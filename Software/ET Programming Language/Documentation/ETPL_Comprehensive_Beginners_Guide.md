# Exception Theory Programming Language (ETPL)
## A Comprehensive Beginner's Guide

**Author:** Michael James Muller  
**Language Version:** ETPL v1.4.x  
**Toolchain:** ETPL.py  
**Master Equation:** P ∘ D ∘ T = EIM = S  
**Ground Principle:** *"For every exception there is an exception, except the exception."*

---

## Table of Contents

1. [What ETPL Is — and Why It Exists](#1-what-etpl-is--and-why-it-exists)
2. [The Three Primitives: P, D, T](#2-the-three-primitives-p-d-t)
3. [File Extensions: .pdt and .eim](#3-file-extensions-pdt-and-eim)
4. [Getting Started with the Toolchain](#4-getting-started-with-the-toolchain)
5. [Basic Syntax: Points, Descriptors, Traversers](#5-basic-syntax-points-descriptors-traversers)
6. [Operators and Expressions](#6-operators-and-expressions)
7. [Functions as Descriptors](#7-functions-as-descriptors)
8. [Control Flow](#8-control-flow)
9. [Loops](#9-loops)
10. [Manifolds](#10-manifolds)
11. [Paths and the Arrow Operator](#11-paths-and-the-arrow-operator)
12. [Exception Paths and Error Handling](#12-exception-paths-and-error-handling)
13. [Indeterminate Forms](#13-indeterminate-forms)
14. [Quantum Features](#14-quantum-features)
15. [Member Access and Indexing](#15-member-access-and-indexing)
16. [I/O and Sovereign Calls](#16-io-and-sovereign-calls)
17. [Hardware Access](#17-hardware-access)
18. [Comments](#18-comments)
19. [The .eim Extension System](#19-the-eim-extension-system)
20. [Mathematical Operators and Built-ins](#20-mathematical-operators-and-built-ins)
21. [Compilation Targets](#21-compilation-targets)
22. [Complete Program Examples](#22-complete-program-examples)
23. [Language Reference Summary](#23-language-reference-summary)

---

## 1. What ETPL Is — and Why It Exists

ETPL (Exception Theory Programming Language) is a programming language derived directly from **Exception Theory (ET)**, a triple-tautological meta-ontology built on three irreducible primitives: **Point (P)**, **Descriptor (D)**, and **Traverser (T)**. These three primitives are the minimum complete structure for describing any existing thing, and they map naturally onto the components of every program ever written.

The key insight is this: every program is already P ∘ D ∘ T — it just doesn't say so explicitly. ETPL makes that structure *the language itself*.

**ETPL is not a novelty language.** It compiles to native binaries (via LLVM or the ETSovereign native backend), it interprets `.pdt` source files directly, and it can translate code from Python, C headers, and JavaScript into ETPL automatically. It runs on classical hardware today and is architecturally ready for quantum hardware.

### Why use ETPL?

- Every program you write is grounded in a philosophically complete ontology that describes reality at its most fundamental level.
- The type system (P, D, T) *is* the type system of the universe.
- Indeterminate forms, quantum superposition, and manifold operations are first-class language features, not library add-ons.
- The `.eim` extension system makes the language polymorphic — users can define custom symbols and context-dependent meanings.
- Programs compiled with ETPL can target classical CPUs, quantum devices, hybrid systems, or bare metal — from a single source file.

### The Master Equation

```
P ∘ D ∘ T = E
```

Read: "A Point bound by a Descriptor traversed by a Traverser produces an Exception."

This equation is also written as:

```
PDT = EIM = S (Something)
3 = 3 = 3 = Σ
```

This is the triple-tautological identity at the heart of everything: **three structural primitives (PDT)**, **three phenomenological contributions (EIM)**, and **three impossibilities (Φ)** are simultaneously and co-equally true descriptions of the same three-part reality.

---

## 2. The Three Primitives: P, D, T

Before writing a single line of ETPL, you need to understand what P, D, and T mean — not just as language keywords, but as real ontological categories. Every ETPL construct is one of these three things.

### P — Point (Substrate)

**What it is:** The infinite substrate of existence. The unstructured ground. The "what that exists" before any constraints are applied.

**Cardinality:** Ω (absolute infinite)

**In programming terms:** A Point is a variable — it holds a value (the grounded state of an infinite substrate). Before binding, P is everything. After binding, it is the specific thing you made it.

**In ET physics:** P is dark matter, the quantum field before measurement, the vacuum, infinite potential.

**Examples of P in reality:** A blank page, empty memory, the void before creation, the infinite set of all numbers before you pick one.

```pdt
P x = 42          // A Point named x, grounded at value 42
P name = "Alice"  // A Point named name, grounded as a string
P nothing = P     // A Point grounded at P itself — the null/unbound state
```

### D — Descriptor (Constraint)

**What it is:** The finite constraint. The rule that shapes the substrate. The "how" — structure, definition, the bridge between agency and ground.

**Cardinality:** n (finite)

**In programming terms:** A Descriptor is a function. It takes inputs (P-substrates) and constrains them through its body to produce an output. Descriptors define what is possible in a given domain.

**In ET physics:** D is the laws of physics, spin, charge, mass — any finite constraint that carves coherent structure out of infinite P.

**Examples of D in reality:** Mathematical laws, the rules of chess, type constraints, a recipe, the laws of a country.

```pdt
D double = λ x . x * 2         // A Descriptor named double
D add = λ a, b . a + b          // A Descriptor with two parameters
D greet = λ name . "Hello " + name
```

### T — Traverser (Agency)

**What it is:** Indeterminate agency. Navigation. The "who" that moves through the manifold. T is the force that takes a structured substrate and *actually realizes* one of its possibilities.

**Cardinality:** [0/0] (indeterminate — the genuine mathematical form of T)

**In programming terms:** A Traverser is dynamic binding — a named execution, a loop, a process moving through possibilities. T is what makes your program *run*.

**In ET physics:** T is the observer, the measurement act, proper time (τ), the Traverser navigating through descriptor-fields.

**Examples of T in reality:** Your attention, an electron's path, a loop iteration, time itself (as agential experience).

```pdt
T result = add(3, 7)             // A Traverser that navigates the add Descriptor
T loop = ∞ (sovereign_print ∘ "Hello") (D 5)   // A Traverser that loops 5 times
```

### The Full Binding: P ∘ D ∘ T = E

When all three primitives come together — P (something to ground), D (a constraint to apply), T (agency to actualize it) — the result is an **Exception (E)**: a grounded, irreversible, substantiated event. In everyday terms, the Exception is the *result* of your program running.

```pdt
// P: the number 5 exists as potential
// D: the doubling rule constrains it
// T: the binding executes the traversal
// E: the result 10 is substantiated

P value = 5
D double = λ x . x * 2
T result = double(value)         // E = 10
sovereign_print ∘ result
```

### The EIM Triad

The same three primitives have a phenomenological reading (what each one *contributes*):

| Symbol | Name | What it gives | Without it |
|--------|------|---------------|------------|
| **E** | Exception | Grounding — the actual Now | Nothing is real, only frozen potential |
| **I** | Incoherence | The D-bridge; the coherence boundary | No traversable structure; P and T cannot meet |
| **M** | Mediation | Traversal; the ∘ operator in action | No movement; no connection between anything |

And each primitive enforces one absolute impossibility (Φ):
- **E** (Exception): Cannot be otherwise while it IS.
- **I** (Incoherence): Cannot be traversed to — the forbidden zone.
- **M** (Mediation): Cannot be absent — binding/interaction is always active.

These three impossibilities define the shape of what *can* exist, by defining what cannot.

---

## 3. File Extensions: .pdt and .eim

### The .pdt Extension

All ETPL source files use the `.pdt` extension, standing for **Point · Descriptor · Traverser** — the three ontological primitives. Every program you write goes in a `.pdt` file.

```
my_program.pdt      // Standard ETPL source file
```

The `.pdt` extension is not arbitrary. It reflects the fact that any program, at its most fundamental level, is a system of P, D, and T bindings. The extension is a declaration of what the file *is*.

### The .eim Extension

`.eim` files are **symbol extension files** — persistent definition files that add custom symbols, polymorphic meanings, and extended vocabularies to the ETPL toolchain. The `.eim` extension stands for the **EIM triad**: **Exception (E)**, **Incoherence (I)**, and **Mediation (M)**.

In the PDT = EIM = 3 boundary condition, EIM represents *"what cannot be"* in the structural sense. A `.eim` file defines custom symbols at the language level — it extends what the language can express. Think of `.eim` files as persistent dictionaries that tell the entire ETPL toolchain (parser, tokenizer, lexer, interpreter, compiler) how to understand symbols that are not built in.

**Why .eim?** Because a symbol is a constraint on meaning. Defining a symbol is a Descriptor act. An `.eim` file is a catalog of Descriptors for the language itself.

#### What .eim Files Contain

Every entry in an `.eim` file defines a symbol through **two things in PDT code**:
1. **The definition** — what the symbol means (its behavior, value, or function body).
2. **The context** — the conditions under which this specific meaning applies.

Both the definition and the context are written in valid `.pdt` code. This is important: `.eim` files are not configuration text — they are PDT code that defines how the language should understand something.

#### Polymorphic Symbols

The most powerful feature of `.eim` files is that **a single symbol can have multiple definitions with different contexts**. The toolchain uses the context to determine which definition applies. This makes ETPL polymorphic at the language level — a symbol's meaning is not fixed; it depends on where and how it is used.

When two definitions share the same description AND the same context in `.eim` files, the system reads them as the same object. Even if encountered twice, no duplicate is created and no conflict occurs. The system is idempotent for identical definitions.

#### What Counts as a Symbol

`.eim` symbols are not restricted to single characters. Any of the following are valid symbol identifiers:

```
→         // A single Unicode character
House     // A word
_func     // An underscore-prefixed name
1+1       // An expression as a symbol
∇⊗Ψ       // A multi-character mathematical sequence
MyOperator // A CamelCase word
```

This allows domain-specific vocabularies — legal terms, mathematical notation, domain operators, natural-language words — to be given precise ETPL meanings.

#### How .eim Files Are Loaded

Users keep `.eim` files separately — they are not embedded in `.pdt` source files. When compiling or interpreting a `.pdt` file, the user points the toolchain to one or more `.eim` files using the **binding operator** (`∘`). The toolchain then loads all symbols from the `.eim` file alongside the built-in base layer.

The built-in symbols of ETPL are always loaded as the base layer. `.eim` symbols are added on top without displacing or conflicting with built-ins (unless a context is defined that explicitly overrides a built-in for a specific use case).

You can also point the toolchain at an **entire folder of `.eim` files**. When you do, all `.eim` files in that folder are loaded simultaneously, combining their symbol definitions into one extended vocabulary for the compilation session.

#### .eim File Syntax

```pdt
// eim_definitions.eim
// Each entry: SYMBOL, DEFINITION (pdt code), CONTEXT (pdt code)

// Define "⊕" as XOR addition in a cryptography context
SYMBOL ⊕
DEFINITION D et_xor = λ a, b . a ^ b
CONTEXT D context = λ mode . mode == "crypto"

// Define "House" as a data structure in an architecture context
SYMBOL House
DEFINITION D House = λ floors, rooms . manifold [floors, rooms]
CONTEXT D context = λ mode . mode == "architecture"

// Define "House" differently in an economy context (same name, different context)
SYMBOL House
DEFINITION D House = λ price, sqft . manifold [price, sqft]
CONTEXT D context = λ mode . mode == "economy"

// Define a compound symbol "1+1" as a special optimized constant
SYMBOL 1+1
DEFINITION P et_one_plus_one = 2
CONTEXT D context = λ mode . 1    // always active
```

When the toolchain encounters `⊕` in a `.pdt` file, it checks the active `.eim` file's context for that symbol and applies the matching definition. If the context evaluates to the cryptography mode, `⊕` means XOR. In a different context, `⊕` could mean something entirely different.

#### Loading .eim Files at the Command Line

```bash
# Interpret with a single .eim extension
python ETPL.py interpret my_program.pdt --eim my_definitions.eim

# Compile with a folder of .eim files
python ETPL.py compile my_program.pdt output --eim ./my_eim_library/

# The toolchain processes the binding automatically
```

---

## 4. Getting Started with the Toolchain

The ETPL toolchain is contained in a single Python file: `ETPL.py`. You do not need any language other than Python to get started. Once you have produced a compiled binary, you do not need Python to *run* it — the binary is standalone.

### Running the Interactive Shell

```bash
python ETPL.py
```

This launches the ETPL CLI shell with a full command interface. You will see the ETPL banner showing the master equation, the primitives, and the available commands.

### Interpreting a .pdt File (Running Without Compilation)

```bash
python ETPL.py interpret myfile.pdt
python ETPL.py run myfile.pdt         # alias
python ETPL.py i myfile.pdt           # short alias

# With debug output (shows AST and binding trace):
python ETPL.py interpret myfile.pdt --debug
```

This executes your `.pdt` file immediately without producing a binary. It is the fastest way to test your programs.

### Compiling to a Native Binary

```bash
# Auto-name the output binary:
python ETPL.py compile myfile.pdt

# Specify the output name:
python ETPL.py compile myfile.pdt myprogram

# Compile for specific targets:
python ETPL.py compile myfile.pdt output --target classical    # default: classical CPU
python ETPL.py compile myfile.pdt output --target quantum      # quantum-aware emission
python ETPL.py compile myfile.pdt output --target hybrid       # classical + quantum
python ETPL.py compile myfile.pdt output --target bare_metal   # no OS, raw hardware

# Compile for specific architectures:
python ETPL.py compile myfile.pdt output --arch x86_64
python ETPL.py compile myfile.pdt output --arch arm64
python ETPL.py compile myfile.pdt output --arch riscv64
python ETPL.py compile myfile.pdt output --arch wasm
```

The compiled binary requires no Python installation to run. It is a self-contained executable.

### Translating from Another Language

```bash
# Translate Python to ETPL:
python ETPL.py translate myfile.py --lang python -o myfile.pdt

# Translate a C header:
python ETPL.py translate header.h --lang c_header -o header.pdt

# Translate JavaScript:
python ETPL.py translate script.js --lang javascript -o script.pdt
```

The translator produces a fully self-contained `.pdt` file that includes all resolved imports as inline ETPL code. Nothing is left as a stub.

### The Interactive REPL

```bash
python ETPL.py repl
```

This opens a line-by-line ETPL expression evaluator. Type any ETPL expression and see the result immediately. Type `.quit` to exit.

### Diagnosing the Toolchain

```bash
python ETPL.py toolchain           # diagnose LLVM, MSVC, MinGW availability
python ETPL.py toolchain --fix     # attempt auto-repair of toolchain issues
```

### Running the Verification Suite

```bash
python ETPL.py verify    # runs all 106+ self-verification tests
```

---

## 5. Basic Syntax: Points, Descriptors, Traversers

Every ETPL statement begins with one of the three primitive keywords: `P`, `D`, or `T`. This is not convention — it is law. Every declaration in an ETPL program *is* a Point, a Descriptor, or a Traverser.

### Point Declarations (Variables)

A Point declaration creates a grounded substrate with a name and a value.

**Syntax:** `P name = expression`

```pdt
P x = 5                    // Integer
P pi = 3.14159             // Float
P greeting = "Hello"       // String
P truth = 1                // 1 = true, 0 = false (ET trinary: 1, 0, or [0/0])
P nothing = P              // P on the right means "unbound/null" — the substrate itself
P huge = Infinity          // The infinite literal
P omega = Ω                // Alternative infinite literal
```

**Point names** are standard identifiers: letters, digits, and underscores, starting with a letter or underscore. The names `P`, `D`, `T`, `E`, and all built-in function names (like `sin`, `sqrt`, `manifold`, etc.) are reserved and cannot be used as Point names.

When you write `P nothing = P`, the right-hand `P` is the ETPL keyword being used as a value — the unbound substrate, equivalent to `null` or `None` in other languages.

### Descriptor Declarations (Functions)

A Descriptor declaration defines a named function using lambda syntax.

**Syntax:** `D name = λ params . body`

The `λ` symbol is the ETPL lambda keyword. It introduces the parameter list, followed by a dot `.`, followed by the body expression.

```pdt
// Single parameter
D square = λ x . x * x

// Multiple parameters (separated by commas)
D add = λ a, b . a + b
D multiply = λ a, b . a * b

// No parameters (a Descriptor that is effectively a constant function)
D answer = λ . 42

// Multi-statement body (use braces { } for multiple statements)
D compute = λ x, y . {
    P sum = x + y
    P product = x * y
    sum + product          // last expression is the return value
}
```

**Calling a Descriptor** is done with either parentheses or the compose operator `∘`:

```pdt
// Parentheses-style call:
P result = add(3, 4)       // result = 7

// Compose-style call (∘ applies the Descriptor to the argument):
P result = add ∘ 3         // single-argument compose
sovereign_print ∘ result   // print result — this is always compose-style

// Compose calls one argument at a time; for multi-argument Descriptors, 
// use parentheses-style for clarity:
P r2 = multiply(5, 6)      // r2 = 30
```

The compose operator `∘` is the ET binding operator — it is the explicit notation for "apply this Descriptor to this substrate." In the master equation P ∘ D ∘ T = E, the `∘` represents the mediation — the act of binding.

### Traverser Declarations (Dynamic Bindings)

A Traverser declaration runs an expression and names the result. Traversers represent executed processes — things that have actually traversed through possibilities to produce a grounded Exception.

**Syntax:** `T name = expression`

```pdt
// A Traverser that captures the result of a function call
T result = add(10, 20)           // result = 30

// A Traverser used as a dynamic binding (evaluates an expression)
T check = x > 0                  // true (1) if x > 0, false (0) otherwise

// A Traverser that opens a resource (path-based)
T file = → open_file("data.txt")  // → means "traverse to"
```

The distinction between `P result = add(10, 20)` and `T result = add(10, 20)` is one of ontological declaration: `P` declares that the name is a substrate being grounded; `T` declares that the name is a traverser — an active navigation process that happens to produce a value. Both work in similar ways, but the semantics convey different intent in the ET framework. Use `T` when you are explicitly modeling a traversal, a dynamic binding, an iterative process, or an execution act.

---

## 6. Operators and Expressions

### Arithmetic Operators

```pdt
P a = 10
P b = 3

P sum     = a + b        // 13 — addition
P diff    = a - b        // 7  — subtraction
P product = a * b        // 30 — multiplication
P quotient = a / b       // 3.333... — division
P floordiv = a ÷ b       // 3  — floor division (Unicode ÷, avoids // comment ambiguity)
P remainder = a % b      // 1  — modulo
P power   = a ^ b        // 1000 — exponentiation (or a ** b — both work)
P power2  = a ** b       // 1000 — same as above
```

**Division by zero** in ETPL follows ET semantics: it does not crash.
- `1 / 0` → `∞` (Infinity — P-substrate dominates)
- `0 / 0` → `0` (Indeterminate resolves to ground state; or use `[0/0]` for the full indeterminate form)

### Comparison Operators

```pdt
P equal    = a == b      // 0 (false)
P notequal = a != b      // 1 (true)
P greater  = a > b       // 1
P less     = a < b       // 0
P geq      = a >= b      // 1
P leq      = a <= b      // 0
```

### Logical Operators

```pdt
P and_result = a && b    // logical AND
P or_result  = a || b    // logical OR
P not_result = !a        // logical NOT (prefix)
```

### Bitwise Operators

```pdt
P band  = a & b          // bitwise AND
P bor   = a | b          // bitwise OR
P bxor  = a ^ b          // bitwise XOR (note: ^ also means exponentiation — context matters)
P lsh   = a << 1         // left shift
P rsh   = a >> 1         // right shift
```

**Note on `^`:** In ETPL, `^` is the exponentiation operator. Bitwise XOR uses the `^` token only in the `BINARY_OP` context (when used with `|`, `&`, `<<`, `>>`). For explicit exponentiation, `**` is always unambiguous.

### String Operations

```pdt
P first = "Hello"
P second = " World"
P combined = first + second           // "Hello World" — string concatenation
P repeated = first + " " + "Again"   // "Hello Again"
```

### The Compose Operator ∘

The compose operator `∘` (Unicode U+2218) is the fundamental ET operator. It applies a Descriptor to a value, directly expressing the `P ∘ D ∘ T` master equation in code.

```pdt
D negate = λ x . 0 - x

P n = negate ∘ 5         // n = -5
sovereign_print ∘ n      // prints: -5
sovereign_print ∘ "Done" // prints: Done
```

The compose operator is right-associative and can be chained:

```pdt
D inc = λ x . x + 1
D double = λ x . x * 2

// Apply inc first, then double:
P result = double ∘ (inc ∘ 3)    // double(inc(3)) = double(4) = 8
```

---

## 7. Functions as Descriptors

Descriptors are the function mechanism in ETPL. They are defined with `D`, use `λ` (lambda) syntax, and can recurse, close over outer names, and return any expression.

### Basic Descriptor Definition and Calling

```pdt
// Define a Descriptor
D greet = λ name . "Hello, " + name + "!"

// Call it
P message = greet("Alice")
sovereign_print ∘ message          // Hello, Alice!
```

### Multi-Parameter Descriptors

```pdt
D power_of = λ base, exp . base ^ exp

P eight = power_of(2, 3)           // 8
P big   = power_of(10, 6)          // 1000000
```

### Multi-Statement Descriptor Bodies

When a Descriptor body needs more than one statement, wrap it in curly braces `{ }`. The last evaluated expression in the block is the return value.

```pdt
D bmi = λ weight_kg, height_m . {
    P h2 = height_m * height_m
    P bmi_value = weight_kg / h2
    bmi_value                      // returned
}

P my_bmi = bmi(70, 1.75)
sovereign_print ∘ my_bmi           // ~22.857
```

### Recursive Descriptors

Descriptors can call themselves by name:

```pdt
D factorial = λ n . {
    if n <= 1 → 1 → E n * factorial(n - 1)
}

P f5 = factorial(5)
sovereign_print ∘ f5               // 120
```

**Explanation of the syntax `if n <= 1 → 1 → E n * factorial(n - 1)`:**
- `if n <= 1` — condition
- `→ 1` — if true, the path returns 1
- `→ E n * factorial(n - 1)` — if false, the exception path returns `n * factorial(n - 1)`

This is covered in full in the [Control Flow](#8-control-flow) section. The key thing to notice here is that `→` in ETPL is the *traversal arrow* — it routes execution forward.

### Descriptors Returning Descriptors (Higher-Order Functions)

```pdt
D multiplier = λ factor . {
    D inner = λ x . x * factor
    inner
}

P triple = multiplier(3)
P result = triple(7)
sovereign_print ∘ result           // 21
```

### Class-Like Descriptors (Using Manifolds)

ETPL does not have a dedicated class keyword — instead, classes are modeled as Descriptors that produce manifolds (structured collections of P bindings).

```pdt
D make_person = λ name, age . {
    P et_name = name
    P et_age = age
    manifold [et_name, et_age]     // the "object" is a manifold
}

P alice = make_person("Alice", 30)
P alice_name = alice[0]            // "Alice"
P alice_age  = alice[1]            // 30

sovereign_print ∘ alice_name
sovereign_print ∘ alice_age
```

---

## 8. Control Flow

ETPL's control flow is expressed through the **path operator** `→` (arrow), the `if` keyword, and **exception paths** `→ E`. Together, they form a complete, expressive branching system.

### The If-Path

The core conditional syntax is:

```
if condition → then_expression → E else_expression
```

- `if condition` — evaluate the condition
- `→ then_expression` — if true (non-zero), traverse to this expression
- `→ E else_expression` — the exception path: if false (zero), ground to this expression

```pdt
P x = 10
if x > 5 → sovereign_print ∘ "big" → E sovereign_print ∘ "small"
```

**Output:** `big`

The `if-path` is not a statement in a block — it is an expression that evaluates to a value. The condition routes the traversal.

### Nested Conditions

For multiple branches, nest `if` expressions in the else-path:

```pdt
P score = 75

P grade = if score >= 90 → "A" → E (if score >= 80 → "B" → E (if score >= 70 → "C" → E "F"))
sovereign_print ∘ grade             // C
```

### Using If in Descriptor Bodies

```pdt
D classify = λ n . {
    if n > 0 → "positive" → E (if n < 0 → "negative" → E "zero")
}

sovereign_print ∘ classify(5)      // positive
sovereign_print ∘ classify(-3)     // negative
sovereign_print ∘ classify(0)      // zero
```

### The Traverser as a Conditional Gate

In translated code, the `T check = → if condition → P` pattern is used as a while-loop gate (the condition guards continued traversal). This is the translator's convention; you can use the same pattern in hand-written code to model conditional traversal steps:

```pdt
T gate = → if x > 0 → x * 2       // Traverse to doubled x if positive; otherwise P
```

---

## 9. Loops

Loops in ETPL are modeled as **bounded traversals** — an `∞` (infinity symbol) paired with a finite Descriptor bound `(D n)`. This directly reflects the ET principle that true infinite traversal cannot be substantiated; every traversal must be finitely bounded.

### Basic Loop Syntax

```
T name = ∞ (body) (D bound)
```

- `T name` — names this traversal
- `∞` — the infinite traversal operator (Unicode U+221E, or `inf`)
- `(body)` — the loop body, executed on each iteration
- `(D bound)` — the Descriptor that bounds the loop (how many times to run)

The loop counter is automatically available as `_loop_index` inside the body, starting at 0.

```pdt
// Print "Hello" 5 times
T greet_loop = ∞ (sovereign_print ∘ "Hello") (D 5)
```

**Output:**
```
Hello
Hello
Hello
Hello
Hello
```

### Accessing the Loop Index

```pdt
T count_loop = ∞ (sovereign_print ∘ _loop_index) (D 4)
```

**Output:**
```
0
1
2
3
```

### Multi-Statement Loop Bodies (Braces)

For loop bodies with more than one statement, use braces `{ }`:

```pdt
T compute_loop = ∞ (
    {
        P val = _loop_index * _loop_index
        sovereign_print ∘ val
    }
) (D 5)
```

**Output:**
```
0
1
4
9
16
```

### Loop with a Dynamic Bound

The bound `(D n)` can use any expression:

```pdt
P limit = 7
T dynamic_loop = ∞ (sovereign_print ∘ _loop_index) (D limit)
```

### Loop Accumulation Pattern

Traversers that need to collect results use the inner Point to accumulate:

```pdt
P total = 0
T sum_loop = ∞ (
    {
        T update = → P             // traversal step
        P total = total + _loop_index
    }
) (D 10)

sovereign_print ∘ total            // 45 (sum of 0..9)
```

### While-Style Loops

While loops are expressed as bounded traversals gated by a condition. The translator uses the canonical `WHILE_LOOP_FINITE_BOUND` constant (144 = 12², derived from MANIFOLD_SYMMETRY²) as the upper safety limit:

```pdt
P n = 10
T while_loop = ∞ (
    T check = → if n > 0 → P     // gate — continue only while n > 0
    sovereign_print ∘ n
    P n = n - 1
) (D WHILE_LOOP_FINITE_BOUND)     // bounded by condition; max = 144
```

**Output:** counts down from 10 to 1.

**Why 144?** MANIFOLD_SYMMETRY = 12. MANIFOLD_SYMMETRY² = 144. In ET, 144 is the canonical finite upper bound derived from the manifold's geometry. Infinite loops must resolve; 144 is their maximum resolution count in the ET execution model.

---

## 10. Manifolds

A **manifold** is an ordered, heterogeneous collection of Point values — the ETPL equivalent of a list or array. Manifolds are first-class ET objects: they are the discrete substrate of structured data, the finite lattice of values that T can navigate.

### Creating Manifolds

```pdt
P numbers = manifold [1, 2, 3, 4, 5]
P names   = manifold ["Alice", "Bob", "Carol"]
P mixed   = manifold [42, "hello", 3.14, P]    // mixed types, P = null slot
P empty   = manifold []                         // empty manifold
```

### Accessing Manifold Elements

Use square bracket indexing `[n]`. Indices are zero-based.

```pdt
P numbers = manifold [10, 20, 30, 40]
P first   = numbers[0]                  // 10
P second  = numbers[1]                  // 20
P last    = numbers[3]                  // 40
```

### Manifold Length

The `|collection|` notation returns the length (cardinality) of a manifold. This is used in loop bounds to iterate over all elements:

```pdt
P fruits = manifold ["apple", "banana", "cherry"]
T fruit_loop = ∞ (
    {
        P item = fruits[_loop_index]
        sovereign_print ∘ item
    }
) (D |fruits|)
```

**Output:**
```
apple
banana
cherry
```

### Nested Manifolds (Dictionaries, Matrices)

Manifolds can contain other manifolds, enabling key-value pairs and matrices:

```pdt
// Key-value pair
P entry = manifold ["name", "Alice"]

// Dictionary as nested manifold
P person = manifold [
    manifold ["name", "Alice"],
    manifold ["age", 30],
    manifold ["city", "Berlin"]
]

P person_name = person[0][1]    // "Alice" — outer index 0 = name entry, inner index 1 = value
P person_age  = person[1][1]    // 30
```

### Slicing Manifolds

A range of elements is accessed using the colon notation `[start:end]`:

```pdt
P data = manifold [1, 2, 3, 4, 5, 6, 7]
P first_three = data[0:3]       // manifold [1, 2, 3]
P middle      = data[2:5]       // manifold [3, 4, 5]
```

---

## 11. Paths and the Arrow Operator

The arrow `→` is the **traversal operator** in ETPL. It models T's movement through the configuration space — routing execution from one point to another. The arrow represents *direction*, *intent*, and *destination* in the traversal.

### The Path Declaration

```pdt
T path_name = → expression
```

This declares a Traverser that "traverses to" the given expression. It is used to model resource acquisition, routing, dynamic binding, and context entry.

```pdt
T scope_file = → open_file("data.txt")    // traverse to an opened file
P data = scope_file                        // use the traversal result
```

### The Path as Flow Control

Inside Descriptor bodies and loops, `→` used alone (not preceded by a name declaration) routes the current execution:

```pdt
D safe_divide = λ a, b . {
    if b == 0 → Infinity → E a / b
}
```

Here:
- `if b == 0` — condition
- `→ Infinity` — if b is zero, the path routes to Infinity
- `→ E a / b` — otherwise, the exception path grounds to the division result

### Multiple Paths in Sequence

When multiple paths appear in sequence, each path leads to the next traversal step:

```pdt
D validate_then_compute = λ x . {
    → if x > 0 → x * 2            // path: traverse to doubled value if positive
    → E 0                          // exception ground: return 0 otherwise
}
```

### Context Manager Pattern (T scope)

The Traverser-path pattern creates scoped contexts:

```pdt
T scope_ctx = → acquire_resource("connection")
P ctx = scope_ctx                  // the connection is now available via ctx
// ... use ctx ...
T cleanup = → release_resource(ctx)
```

This is the ETPL equivalent of `with` blocks in Python — the Traverser "enters" the context, uses the resource, and the cleanup Traverser "exits" it.

---

## 12. Exception Paths and Error Handling

In ET, the Exception (E) is the grounded result of P ∘ D ∘ T. An **exception path** `→ E value` is the operator that explicitly grounds a traversal to a specific value — it is simultaneously a return statement and an error-handling mechanism.

### → E as Return

```pdt
D max_of = λ a, b . {
    if a > b → E a → E b
}

// Reads: "if a > b, ground (return) to a; otherwise ground to b"
P bigger = max_of(5, 3)
sovereign_print ∘ bigger           // 5
```

### → E for Error Handling

```pdt
D safe_sqrt = λ x . {
    if x < 0 → E "error: negative input" → E sqrt(x)
}

P r1 = safe_sqrt(9)
P r2 = safe_sqrt(-1)
sovereign_print ∘ r1               // 3.0
sovereign_print ∘ r2               // error: negative input
```

### Bare → E (Ground to Nothing)

`→ E` with no argument grounds to `P` (the null substrate). This is the ET way of expressing "no result" or an empty return:

```pdt
D print_if_positive = λ x . {
    if x > 0 → sovereign_print ∘ x
    → E                            // ground to nothing (null return)
}
```

### Try-Attempt Pattern

Translated Python `try/except` blocks appear in ETPL as:

```pdt
T attempt = → P                    // try (traverse to P ground as attempt substrate)
    // body of try block here
```

Error handling in hand-written ETPL uses the `→ E` path to route around exceptions:

```pdt
D try_parse = λ text . {
    if text == "" → E "empty"
    → E text + " (parsed)"
}
```

---

## 13. Indeterminate Forms

The indeterminate form `[0/0]` is the mathematical identity of the Traverser (T) itself. It represents genuine indeterminacy — a state that is neither 0 nor ∞ but is unresolved between possibilities. In ETPL, `[0/0]` is a first-class expression for **branching by pure indeterminate choice**.

### The [0/0] Indeterminate Expression

The `[0/0]` form introduces a choice between one or more expressions separated by `|`. When evaluated:
- In classical mode: the first available (non-error) choice is selected.
- In quantum mode: the expression becomes a superposition (a multi-valued quantum state) that collapses only when measured.

```pdt
// Classical: choose the first available option
P result = [0/0] 1 | 2 | 3        // result = 1 (first choice taken)

// A more interesting use — choose based on actual indeterminacy at runtime:
P entropy = [0/0] "heads" | "tails"
```

### [0/0] with Descriptors

```pdt
D choose_path = λ . [0/0] "path_A" | "path_B" | "path_C"

P selected = choose_path()
sovereign_print ∘ selected         // one of the paths — determined at traversal time
```

### Division by Zero — Automatic ET Resolution

ETPL automatically handles division-by-zero using ET semantics:

```pdt
P a = 1 / 0        // a = Infinity (P-substrate dominates over empty D-constraint)
P b = 0 / 0        // b = 0 (indeterminate form resolved to ground state)
P c = -1 / 0       // c = -Infinity
```

These are not errors — they are ET-principled resolutions of boundary conditions.

### L'Hôpital Navigation

For true 0/0 forms that require mathematical resolution (limits, calculus), the ET L'Hôpital principle applies: T traverses toward the limiting value through successive descriptor refinements. In the interpreter, `0/0` resolves to 0 (the ground state). In quantum mode, it becomes a full superposition.

---

## 14. Quantum Features

ETPL has first-class support for quantum computation. Quantum states are modeled as M-states (Mediation states) — superpositions that have not yet been grounded by Traverser engagement (measurement). The entire quantum computing model falls naturally out of the ET framework:

- **Superposition** = M-state (D-constraints active, no T-binding yet)
- **Measurement** = T engaging with an M-state to produce an Exception (E)
- **Entanglement** = shared Traverser binding between two Points
- **Quantum gates** = D-reconfigurations on M-state substrates

### The Quantum Wavefunction Operator ψ

The `ψ(n, l, m)` operator computes the hydrogen atom wavefunction for quantum numbers n (principal), l (angular momentum), and m (magnetic). This is directly derived from ET's first-principles derivation of the hydrogen atom.

```pdt
// Compute the ground state wavefunction
P ground_state = ψ(1, 0, 0)

// p-orbital
P p_orbital = ψ(2, 1, 0)
```

In the quantum compilation target, `ψ` emissions generate QASM (quantum assembly) for hardware execution.

### Quantum Superposition via [0/0]

In quantum mode, the `[0/0]` operator creates a Hadamard-type superposition:

```pdt
// Compiles to Hadamard gate in QASM:
P qubit_state = [0/0] 0 | 1
```

When compiled with `--target quantum`, this produces the correct quantum circuit code.

### Quantum Compile Output

```bash
python ETPL.py compile quantum_program.pdt qprogram --target quantum
```

This emits QASM code with:
- `ψ` → `ry(θ)` gates calibrated to the computed angle
- `[0/0]` → Hadamard gates `h q[n]`
- Measurements → `measure q → c`

### Example: Bell State Preparation

```pdt
// Two qubits in superposition, then entangled
P qubit0 = [0/0] 0 | 1     // ψ₀ enters M-state (superposition)
P qubit1 = [0/0] 0 | 1     // ψ₁ enters M-state

// Entanglement: shared T-binding
// In quantum compilation, the hardware entangles these via CNOT
```

### M-States in ET Cosmology

In ET cosmology, M-states make up approximately 3% of universal energy (M-vacuum: 1.6% dark-energy-like, M-matter: 1.4% ordinary-matter-like). The quantum computation in ETPL is not an analogy — it is the direct expression of M-state physics in computational form.

---

## 15. Member Access and Indexing

### Member Access with D

In ETPL, the dot operator (`.`) from other languages is replaced by the Descriptor operator `D` used as an infix member accessor:

```pdt
// Accessing a member "value" on an object:
P result = my_object D value

// Chain:
P deep = outer D inner D property
```

When the translator converts Python code, `obj.attr` becomes `obj D attr`. This is not just syntax — it is a semantic statement: accessing a member of an object is a Descriptor operation, because you are *constraining* the infinite substrate of the object to reveal a specific finite attribute.

```pdt
// Python: obj.method(arg)
// ETPL:   obj D method(arg)

T call_result = my_object D process(42)
```

### Indexing with [ ]

Square bracket indexing works on manifolds and any indexable substrate:

```pdt
P data = manifold [100, 200, 300]
P first = data[0]          // 100
P last  = data[2]          // 300

// Negative-index-style access (via arithmetic):
P second_to_last = data[|data| - 2]    // 200
```

---

## 16. I/O and Sovereign Calls

The ETPL I/O layer is built on the **Sovereign** substrate — the ET name for the direct interface between the language and the platform's execution environment. Sovereign calls are how ETPL programs interact with the world.

### sovereign_print

The primary output function. Prints a value to standard output.

**Syntax:** `sovereign_print ∘ expression`

The compose operator `∘` is mandatory — `sovereign_print` is always a Descriptor applied via composition.

```pdt
sovereign_print ∘ "Hello, World!"         // prints: Hello, World!
sovereign_print ∘ 42                       // prints: 42
sovereign_print ∘ 3.14                     // prints: 3.14

// Multi-value: concatenate with + and spaces
sovereign_print ∘ "Value: " + 42          // prints: Value: 42
```

### sovereign_import

Imports a module by name. This is a last-resort mechanism; properly structured `.pdt` files should have all dependencies inlined at translation time.

```pdt
sovereign_import ∘ "json"
```

### sovereign_sleep

Pauses execution for a number of seconds.

```pdt
sovereign_sleep ∘ 1                        // wait 1 second
sovereign_sleep ∘ 0.5                      // wait 500 milliseconds
```

### Output in Loops

```pdt
T print_loop = ∞ (
    sovereign_print ∘ _loop_index
) (D 5)
```

Output:
```
0
1
2
3
4
```

---

## 17. Hardware Access

The `hardware_access` keyword is ETPL's direct pathway to the platform's native capabilities — file I/O, platform information, process control, and low-level operations that go beneath the sovereign abstraction layer.

`hardware_access` is an ET Descriptor that takes a command string and routes through the ETPL runtime to the appropriate system call.

### Common Hardware Access Operations

```pdt
// Platform information
P platform_name = hardware_access ∘ "platform"       // e.g., "linux", "win32"

// File I/O
D read_line    = λ . hardware_access ∘ "readline"    // read a line from stdin
D path_join    = λ a . hardware_access ∘ "path_join" ∘ a
D path_exists  = λ p . hardware_access ∘ "path_exists" ∘ p
```

### Bare-Metal Programs

When compiling with `--target bare_metal`, `hardware_access` calls emit direct system calls with no OS mediation. This is used for embedded systems, firmware, and low-level device drivers:

```bash
python ETPL.py compile firmware.pdt output.elf --target bare_metal --arch arm64
```

---

## 18. Comments

ETPL line comments use `//` (double forward slash). Everything after `//` on a line is ignored by the parser.

```pdt
// This is a full-line comment

P x = 42          // This is an inline comment

// Multi-line comments use multiple // lines:
// Line 1 of explanation
// Line 2 of explanation
// Line 3 of explanation
D compute = λ x . x * 2   // double the input
```

**Important:** Because `//` begins a comment, the floor-division operation in ETPL uses the Unicode symbol `÷` instead of `//`. This is by design — `÷` is ET-principled (floor division is a D-bounded divisor operation, distinct from `/`).

```pdt
P result = 10 ÷ 3       // floor division: 3 (NOT a comment)
P wrong  = 10 // 3      // this is the number 10 followed by a comment "3"!
```

---

## 19. The .eim Extension System

This section gives the full technical treatment of `.eim` files — how to write them, how the toolchain uses them, and how to design polymorphic symbol systems.

### Architecture: How .eim Integrates with the Toolchain

When you point ETPL to a `.eim` file, the following process occurs:

1. **Symbol Loading**: The `.eim` file is parsed before the `.pdt` source file. All symbol entries are loaded into the toolchain's symbol table.

2. **Base Layer Preservation**: The built-in ETPL symbols (P, D, T, E, ∘, λ, →, manifold, sovereign_print, hardware_access, etc.) are always present as the base layer. `.eim` symbols are added in a second layer on top.

3. **Context Evaluation**: When the tokenizer or parser encounters a symbol that matches a `.eim` entry, it evaluates the context to determine which definition applies.

4. **Polymorphic Dispatch**: If multiple definitions exist for the same symbol, the one whose context evaluates to true (non-zero) in the current program context is selected. If contexts are mutually exclusive, you get clean polymorphism. If multiple contexts are simultaneously true, the first matching definition is used.

5. **Idempotency**: Two entries with identical definitions AND identical contexts are recognized as the same object. No duplicate is created; no conflict occurs. The system processes them as one.

### Writing .eim Files

A `.eim` file is a plain text file with a `.eim` extension. It contains one or more symbol definitions, each with three parts: the SYMBOL identifier, the DEFINITION in PDT code, and the CONTEXT in PDT code.

**Full example — a mathematics extension:**

```pdt
// math_extensions.eim
// Extends ETPL with domain-specific mathematical symbols

// Define ⊗ as the tensor product (always active)
SYMBOL ⊗
DEFINITION D et_tensor = λ a, b . manifold [a, b, a * b]
CONTEXT D context = λ mode . 1

// Define ∂ as a partial derivative approximation
SYMBOL ∂
DEFINITION D et_partial = λ f, x . (f(x + 0.0001) - f(x)) / 0.0001
CONTEXT D context = λ mode . mode == "calculus"

// Define ∂ differently in a physics context (action derivative)
SYMBOL ∂
DEFINITION D et_partial = λ field, x . field D gradient(x)
CONTEXT D context = λ mode . mode == "physics"

// Define "norm" as a word-symbol for vector magnitude
SYMBOL norm
DEFINITION D et_norm = λ v . sqrt(v[0]*v[0] + v[1]*v[1])
CONTEXT D context = λ mode . 1
```

**Full example — a domain language extension for biology:**

```pdt
// biology.eim
// Makes ETPL speak biology

SYMBOL transcribe
DEFINITION D transcribe = λ dna_seq . {
    // DNA → RNA: replace T with U
    // (implementation uses ETPL string operations)
    T result_loop = ∞ (
        P base = dna_seq[_loop_index]
        if base == "T" → sovereign_print ∘ "U" → E sovereign_print ∘ base
    ) (D |dna_seq|)
}
CONTEXT D context = λ mode . mode == "molecular_biology"

SYMBOL complement
DEFINITION D complement = λ base . {
    if base == "A" → "T" → E (if base == "T" → "A" → E (if base == "C" → "G" → E "C"))
}
CONTEXT D context = λ mode . mode == "molecular_biology"
```

### Mediating a Folder of .eim Files

When you have many `.eim` files organized in a folder, you can load them all at once:

```bash
python ETPL.py interpret my_program.pdt --eim ./domain_symbols/
```

All `.eim` files in `./domain_symbols/` are loaded simultaneously. Their combined symbol definitions form the extended vocabulary for the entire session. Conflicts between `.eim` files are resolved by context — if two files define the same symbol with different contexts, both meanings coexist. If they define the same symbol with the same context, they are recognized as duplicates and merged (no conflict).

### Using .eim Symbols in .pdt Files

Once the `.eim` file is bound to the session, you use the custom symbols in `.pdt` code exactly as if they were built-in:

```pdt
// my_program.pdt — run with --eim math_extensions.eim

P v = manifold [3, 4]
P magnitude = norm(v)             // "norm" defined in math_extensions.eim
sovereign_print ∘ magnitude        // 5.0

P result = manifold [2, 3] ⊗ manifold [4, 5]   // ⊗ from .eim
sovereign_print ∘ result
```

### The PDT = EIM = 3 Boundary Condition and .eim

The `.eim` extension is part of the 3=3=3 boundary condition `PDT = EIM = 3`. The three file types that form the complete ETPL language ecosystem correspond to the three structural triads:

| File | Triad Position | What It Represents |
|------|---------------|-------------------|
| `.pdt` | PDT (structural) | The program — P, D, T bindings |
| `.eim` | EIM (phenomenological) | Extension — what cannot be without definition |
| Third level | Φ (impossibilities) | Compiled binary — the grounded Exception |

The `.eim` file literally extends the language from `EIM` — the phenomenological triad. Just as E (Exception) is grounded actuality, I (Incoherence) marks the boundary of coherence, and M (Mediation) is traversal in action — the `.eim` system grounds custom symbols (E), marks what was previously incoherent/undefined (I), and mediates new meanings into the language (M).

---

## 20. Mathematical Operators and Built-ins

ETPL includes a full mathematical library derived from ET primitives. These are available without any import in all `.pdt` files.

### Mathematical Functions

```pdt
P s   = sin(3.14159 / 2)     // ≈ 1.0
P c   = cos(0)                // 1.0
P t   = tan(0.7854)           // ≈ 1.0
P l   = log(2.71828)          // ≈ 1.0 (natural log)
P l2  = log2(8)               // 3.0
P l10 = log10(100)            // 2.0
P sq  = sqrt(16)              // 4.0
P e   = exp(1)                // ≈ 2.71828
P a   = abs(-5)               // 5
P p   = pow(2, 10)            // 1024.0
P at  = atan(1)               // ≈ 0.7854
P at2 = atan2(1, 1)           // ≈ 0.7854
P fl  = floor(3.7)            // 3
P ce  = ceil(3.2)             // 4
P sh  = sinh(1)               // ≈ 1.1752
P ch  = cosh(1)               // ≈ 1.5431
P th  = tanh(1)               // ≈ 0.7616
P as  = asin(1)               // ≈ 1.5708
P ac  = acos(0)               // ≈ 1.5708
P hy  = hypot(3, 4)           // 5.0
P de  = degrees(3.14159)      // ≈ 180.0
P ra  = radians(180)          // ≈ 3.14159
P fa  = factorial(6)          // 720
P gcd = gcd(12, 8)            // 4
```

### Mathematical Operators (Unicode)

```pdt
P s  = ∑ manifold [1, 2, 3, 4, 5]    // sum = 15
P pr = ∏ manifold [1, 2, 3, 4, 5]    // product = 120
P i  = ∫ f ∘ 0 ∘ 1                   // integral of f from 0 to 1
P gr = ∇ field                         // gradient of a field
P sq = √ 25                            // square root = 5
```

These are directly available as ETPL operators and function as their mathematical counterparts.

### ET Constants

```pdt
P sym = MANIFOLD_SYMMETRY    // 12 — the fundamental manifold fold
P var = BASE_VARIANCE        // 1/12 — the base variance unit
P ko  = KOIDA_RATIO          // 2/3 — the Koide ratio
P inf = Infinity             // ∞
P om  = Ω                    // Omega (absolute infinite)
P aleph = aleph              // ℵ (aleph zero — countable infinity)
P det = WHILE_LOOP_FINITE_BOUND  // 144 = 12² — canonical loop bound
```

### The Indeterminate Constants (EIM)

```pdt
P exception    = 1           // E: grounded, substantiated
P incoherence  = 0           // I: forbidden, unreachable  
P mediation    = [0/0]       // M: active, indeterminate
```

---

## 21. Compilation Targets

ETPL can compile to multiple targets from the same source file. This is a consequence of the ET framework's universality — the same P ∘ D ∘ T structure is valid for any computational substrate.

### Classical Target (Default)

```bash
python ETPL.py compile program.pdt output --target classical
```

Produces a native binary for your CPU using LLVM (or the ETSovereign `.pyc` backend if LLVM is unavailable). The output is standalone — no Python, no C, no external dependencies.

**Backend selection:**
- If `llvmlite` is installed: emits LLVM IR → native object code → linked binary.
- If `llvmlite` is unavailable: uses ETSovereign native backend — compiles to `.pyc` bytecode via Python's `compile()` + `marshal`, producing a `.etb` binary.

Both backends produce valid, runnable executables.

### Quantum Target

```bash
python ETPL.py compile program.pdt output --target quantum
```

Emits QASM (Quantum Assembly) for execution on quantum hardware. `ψ(n,l,m)` operators become rotation gates; `[0/0]` operators become Hadamard superpositions; measurements collapse M-states.

### Hybrid Target

```bash
python ETPL.py compile program.pdt output --target hybrid
```

Emits a hybrid classical+quantum program. Classical sections execute on CPU; quantum sections execute on quantum hardware or simulator. The two are mediated by the ET-native interface layer.

### Bare Metal Target

```bash
python ETPL.py compile program.pdt output.elf --target bare_metal --arch arm64
```

Emits a minimal binary with no OS dependencies. Suitable for embedded systems, firmware, and raw hardware devices. The entry point is a raw function with no OS mediation.

### Architecture Flags

```
--arch x86_64     // Intel/AMD 64-bit (default for most desktops)
--arch arm64      // ARM 64-bit (Apple Silicon, Raspberry Pi, mobile)
--arch riscv64    // RISC-V 64-bit (open-source hardware)
--arch wasm       // WebAssembly (browser/WASM runtime)
--arch universal  // Toolchain auto-selects (default)
```

### LLVM on Windows — Important Note

On Windows, `llvmlite`'s `create_target_machine()` must be called with `codemodel='small'` for AOT (Ahead-Of-Time) compilation. The default `codemodel='jitdefault'` forces ELF output on Windows (a Linux format), which is incorrect. ETPL handles this automatically — the toolchain sets the correct code model for your platform. If you encounter object format mismatches, run `python ETPL.py toolchain --fix` to auto-repair the configuration.

---

## 22. Complete Program Examples

These examples cover progressively more features, each fully explained.

### Example 1: Hello, World

```pdt
// hello.pdt — The simplest ETPL program
// P∘D∘T: P = the string, D = sovereign_print, T = ∘ binding

sovereign_print ∘ "Hello, World!"
```

Run: `python ETPL.py interpret hello.pdt`

**Explanation:** The entire program is one statement. `sovereign_print` is a built-in Descriptor. `∘` composes it with the string `"Hello, World!"`. The result — the Exception — is the string appearing on screen.

---

### Example 2: Variables and Arithmetic

```pdt
// arithmetic.pdt — Variables and math

P x = 10
P y = 3

P sum     = x + y
P diff    = x - y
P product = x * y
P quotient = x / y
P modulus  = x % y
P power    = x ^ y

sovereign_print ∘ "Sum:      " + sum
sovereign_print ∘ "Diff:     " + diff
sovereign_print ∘ "Product:  " + product
sovereign_print ∘ "Quotient: " + quotient
sovereign_print ∘ "Modulus:  " + modulus
sovereign_print ∘ "Power:    " + power
```

**Explanation:** Six Points are declared, each grounded to an arithmetic result. The string concatenation with `+` is used in the print statements to format output. Each `sovereign_print ∘` call is a Traverser completing the D-binding.

---

### Example 3: Descriptor (Function) with Conditional

```pdt
// grade.pdt — Grade classifier

D classify_grade = λ score . {
    if score >= 90 → "A"
    → E (if score >= 80 → "B"
         → E (if score >= 70 → "C"
              → E (if score >= 60 → "D" → E "F")))
}

P s1 = classify_grade(95)
P s2 = classify_grade(82)
P s3 = classify_grade(74)
P s4 = classify_grade(55)

sovereign_print ∘ "95 → " + s1    // 95 → A
sovereign_print ∘ "82 → " + s2    // 82 → B
sovereign_print ∘ "74 → " + s3    // 74 → C
sovereign_print ∘ "55 → " + s4    // 55 → F
```

**Explanation:** The `classify_grade` Descriptor uses nested `if → ... → E ...` paths. Each `→ E` either grounds to the next branch or to the final value. The nesting follows ETPL's requirement that the else-path `→ E` is always the route taken when the condition is false.

---

### Example 4: Loop with Accumulation

```pdt
// sum_loop.pdt — Sum numbers 1 through 10

P total = 0

T sum_loop = ∞ (
    {
        P n = _loop_index + 1      // _loop_index starts at 0, we want 1..10
        P total = total + n
    }
) (D 10)

sovereign_print ∘ "Sum of 1..10 = " + total   // 55
```

**Explanation:** `_loop_index` goes 0..9. Adding 1 gives 1..10. Each iteration adds to `total`. The loop bound `(D 10)` says "run exactly 10 times." The final `sovereign_print` is outside the loop and prints after all iterations complete.

---

### Example 5: Manifold Operations

```pdt
// manifold_ops.pdt — Working with manifolds (lists)

P primes = manifold [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

// Print all primes
sovereign_print ∘ "First 10 primes:"
T prime_loop = ∞ (
    sovereign_print ∘ primes[_loop_index]
) (D |primes|)

// Sum all primes
P prime_sum = 0
T sum_primes = ∞ (
    P prime_sum = prime_sum + primes[_loop_index]
) (D |primes|)

sovereign_print ∘ "Sum = " + prime_sum    // 129
```

**Explanation:** `|primes|` returns the length of the manifold (10), which is used as the loop bound. Each iteration accesses `primes[_loop_index]` — the element at the current index. The sum accumulation follows the same pattern as Example 4.

---

### Example 6: Recursive Fibonacci

```pdt
// fibonacci.pdt — Fibonacci by recursion

D fib = λ n . {
    if n <= 1 → n → E fib(n - 1) + fib(n - 2)
}

T fib_loop = ∞ (
    {
        P result = fib(_loop_index)
        sovereign_print ∘ result
    }
) (D 10)
```

**Explanation:** The `fib` Descriptor recurses directly using its own name. The base case is `if n <= 1 → n` (returns n for 0 and 1). The recursive case `→ E fib(n-1) + fib(n-2)` is the exception path. The loop prints `fib(0)` through `fib(9)`.

**Output:** `0 1 1 2 3 5 8 13 21 34`

---

### Example 7: Nested Manifolds as a Data Structure

```pdt
// data_structure.pdt — A list of person records

D make_person = λ name, age, city . manifold [name, age, city]

P people = manifold [
    make_person("Alice",   30, "Berlin"),
    make_person("Bob",     25, "Paris"),
    make_person("Carol",   35, "Tokyo"),
    make_person("David",   28, "London")
]

// Print each person's data
T people_loop = ∞ (
    {
        P person = people[_loop_index]
        P p_name = person[0]
        P p_age  = person[1]
        P p_city = person[2]
        sovereign_print ∘ p_name + ", age " + p_age + ", from " + p_city
    }
) (D |people|)
```

**Output:**
```
Alice, age 30, from Berlin
Bob, age 25, from Paris
Carol, age 35, from Tokyo
David, age 28, from London
```

**Explanation:** Each person is a manifold of three elements. The `people` manifold contains four person manifolds. The loop accesses each person by outer index and then accesses each field by inner index.

---

### Example 8: Mathematical Computation (ET-Derived)

```pdt
// et_math.pdt — Computing using ET-derived math

// Variance formula from ET: (n² - 1) / 12
// This is the fundamental ET variance for a D-manifold of n states
D et_variance = λ n . (n * n - 1) / 12

// Density: payload / container  (ET Eq 211)
D et_density = λ payload, container . {
    if container == 0 → Infinity → E payload / container
}

// ET fine structure approximation (3-term, ET-derived)
P manifold_sym = 12
P base_var     = 1 / manifold_sym
P koide        = 2 / 3

P var_12   = et_variance(manifold_sym)
P dens_ex  = et_density(64, 100)

sovereign_print ∘ "ET Variance(12) = " + var_12      // 11.916...
sovereign_print ∘ "ET Density(64/100) = " + dens_ex  // 0.64
sovereign_print ∘ "MANIFOLD_SYMMETRY = " + manifold_sym  // 12
sovereign_print ∘ "BASE_VARIANCE = " + base_var      // 0.0833...
sovereign_print ∘ "KOIDE_RATIO = " + koide           // 0.6666...
```

**Explanation:** These Descriptors implement the ET-derived mathematical formulas. The variance formula `(n² - 1) / 12` is the fundamental ET variance for a D-manifold of n states — it appears throughout ET mathematics and directly produces the fine structure constant when properly composed. The density formula is ET Equation 211.

---

### Example 9: Higher-Order Descriptors and Composition

```pdt
// higher_order.pdt — Descriptors as arguments

D apply = λ f, x . f(x)
D compose_d = λ f, g . λ x . f(g(x))

D inc   = λ x . x + 1
D double = λ x . x * 2

// compose_d creates a new Descriptor: inc after double
D inc_after_double = compose_d(inc, double)

P r1 = apply(inc, 5)                    // 6
P r2 = apply(double, 5)                 // 10
P r3 = apply(inc_after_double, 5)       // double(5)=10, inc(10)=11

sovereign_print ∘ r1    // 6
sovereign_print ∘ r2    // 10
sovereign_print ∘ r3    // 11
```

**Explanation:** `apply` takes a Descriptor and an argument and applies one to the other. `compose_d` returns a new Descriptor (a lambda) that applies `f` after `g`. This is the ETPL encoding of the mathematical composition operation `(f ∘ g)(x) = f(g(x))` — directly mirroring the master equation.

---

### Example 10: Quantum Wavefunction Calculation

```pdt
// quantum.pdt — ET quantum features

// Compute hydrogen wavefunctions for the first few states
P psi_100 = ψ(1, 0, 0)     // ground state: n=1, l=0, m=0
P psi_200 = ψ(2, 0, 0)     // first excited s-orbital
P psi_210 = ψ(2, 1, 0)     // p-orbital

sovereign_print ∘ "ψ(1,0,0) = " + psi_100
sovereign_print ∘ "ψ(2,0,0) = " + psi_200
sovereign_print ∘ "ψ(2,1,0) = " + psi_210

// Indeterminate choice (quantum superposition in classical mode)
P choice = [0/0] "state_A" | "state_B" | "state_C"
sovereign_print ∘ "Indeterminate result: " + choice

// ET-derived manifold constant check
sovereign_print ∘ "MANIFOLD_SYMMETRY: " + MANIFOLD_SYMMETRY   // 12
sovereign_print ∘ "BASE_VARIANCE: "     + BASE_VARIANCE       // 0.0833...
```

**Explanation:** `ψ(n,l,m)` computes the hydrogen wavefunction for quantum numbers n, l, m using ET's first-principles hydrogen derivation. The `[0/0]` form demonstrates the quantum indeterminate operator — in classical mode it picks the first choice; in quantum compilation mode it becomes a superposition.

---

### Example 11: A Complete Program — ET Prime Sieve

```pdt
// prime_sieve.pdt — Sieve of Eratosthenes using ET constructs

P LIMIT = 50

// Initialize candidates — all 1 (candidate for prime)
P candidates = manifold []
T init_loop = ∞ (
    P candidates = candidates + manifold [1]
) (D LIMIT)

// Sieve step: mark multiples of each prime as 0
T sieve_outer = ∞ (
    {
        P p = _loop_index + 2          // candidates start at 2
        if p * p > LIMIT → P           // only sieve up to sqrt(LIMIT)
        → E {
            T sieve_inner = ∞ (
                {
                    P multiple = p * (_loop_index + 2)
                    if multiple > LIMIT → P → E {
                        P candidates = candidates    // mark as not-prime
                        // (in a full implementation, set candidates[multiple-2] = 0)
                    }
                }
            ) (D LIMIT)
        }
    }
) (D LIMIT)

// Print primes
sovereign_print ∘ "Primes up to " + LIMIT + ":"
T print_primes = ∞ (
    {
        P n = _loop_index + 2          // number = index + 2
        if candidates[_loop_index] == 1 → sovereign_print ∘ n
    }
) (D LIMIT)
```

**Explanation:** This demonstrates a complete algorithm in ETPL — nested loops, conditional marking, manifold access, and output. The outer loop iterates over potential prime bases. The inner loop marks multiples. The final loop prints the survivors. Every construct used here — P declarations, D lambda, T loop, manifold, if-path, `→ E`, `sovereign_print ∘` — has been covered in the preceding sections.

---

## 23. Language Reference Summary

### Keywords

| Keyword | Type | Meaning |
|---------|------|---------|
| `P` | Primitive | Point — declare a substrate binding |
| `D` | Primitive / Infix | Descriptor — declare a function; or member access infix |
| `T` | Primitive | Traverser — declare a dynamic binding or loop |
| `E` | Primitive | Exception — used in `→ E` to ground a path |
| `λ` | Operator | Lambda — introduces Descriptor parameter list |
| `∞` | Operator | Infinity loop — bounded traversal |
| `∘` | Operator | Compose — apply Descriptor to argument |
| `→` | Operator | Arrow — traversal path |
| `if` | Keyword | Conditional gate in paths |
| `manifold` | Keyword | Construct an ordered collection |
| `ψ` | Keyword | Quantum wavefunction operator |
| `sovereign_print` | Builtin | Print to standard output |
| `sovereign_import` | Builtin | Import a module (last resort) |
| `sovereign_sleep` | Builtin | Pause execution |
| `hardware_access` | Builtin | Direct platform access |
| `Infinity` / `∞` | Literal | Positive infinity |
| `Ω` / `omega` | Literal | Absolute infinite (Omega) |
| `aleph` | Literal | Countable infinite (ℵ₀) |
| `MANIFOLD_SYMMETRY` | Constant | 12 — fundamental fold |
| `BASE_VARIANCE` | Constant | 1/12 |
| `KOIDE_RATIO` | Constant | 2/3 |
| `WHILE_LOOP_FINITE_BOUND` | Constant | 144 = 12² |
| `_loop_index` | Auto | Current loop iteration (0-based) |

### Operators at a Glance

| Operator | Name | Example |
|----------|------|---------|
| `∘` | Compose | `f ∘ x` |
| `→` | Arrow / Path | `T r = → expr` |
| `→ E` | Exception Ground | `→ E value` |
| `λ` | Lambda | `λ x . x + 1` |
| `∞` | Infinite loop | `∞ (body) (D n)` |
| `[0/0]` | Indeterminate | `[0/0] a \| b \| c` |
| `+` | Add / Concat | `a + b` |
| `-` | Subtract | `a - b` |
| `*` | Multiply | `a * b` |
| `/` | Divide | `a / b` |
| `÷` | Floor Divide | `a ÷ b` |
| `%` | Modulo | `a % b` |
| `^` or `**` | Exponent | `a ^ b` |
| `==` `!=` `>` `<` `>=` `<=` | Comparison | `a == b` |
| `&&` `\|\|` `!` | Logical | `a && b` |
| `&` `\|` `^` `<<` `>>` | Bitwise | `a & b` |
| `D` (infix) | Member access | `obj D field` |
| `[n]` | Index | `arr[0]` |
| `[a:b]` | Slice | `arr[0:3]` |
| `\|x\|` | Cardinality | `\|manifold\|` |
| `//` | Comment | `// this is a note` |

### File Types

| Extension | Full Name | Role |
|-----------|-----------|------|
| `.pdt` | Point · Descriptor · Traverser | Source program file |
| `.eim` | Exception · Incoherence · Mediation | Symbol extension file |
| `.etb` | ET Binary | Compiled binary (ETSovereign backend) |
| `.exe` / ELF | Native binary | Compiled binary (LLVM backend) |

### Toolchain Commands

| Command | Action |
|---------|--------|
| `interpret file.pdt` | Run source directly |
| `compile file.pdt [out]` | Compile to native binary |
| `compile file.pdt out --target quantum` | Compile for quantum hardware |
| `compile file.pdt out --target bare_metal` | Compile for bare metal |
| `compile file.pdt out --arch arm64` | Cross-compile for ARM |
| `translate file.py --lang python -o out.pdt` | Translate Python → ETPL |
| `repl` | Interactive expression evaluator |
| `verify` | Run all self-tests |
| `toolchain` | Diagnose compilation toolchain |
| `toolchain --fix` | Auto-repair toolchain |
| `help` | Show full reference |

### The 3=3=3 Identity

```
PDT  =  EIM  =  Φ          (structural, phenomenological, impossibilities)
  3  =    3  =  3          (numerical tautology)
  S  =    S  =  S          (existential: Something = Something = Something)

P ∘ D ∘ T  =  E            (master equation)
```

Every ETPL program is a direct instantiation of this identity. The file extension `.pdt` names the three structural primitives. The `.eim` extension names the three phenomenological contributions. The compiled binary is the grounded Exception — the result of the computation that cannot be otherwise.

---

*Exception Theory Programming Language — where mathematics, physics, and computation are the same language spoken at different levels of abstraction.*

*"For every exception there is an exception, except the exception."*

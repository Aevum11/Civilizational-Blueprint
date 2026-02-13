# ETPL IDE - Complete Setup & Usage Guide

## Quick Setup

### Prerequisites

Ensure you have Python 3.8+ and install required packages:

```bash
pip install PyQt5 llvmlite capstone pefile psutil --break-system-packages
```

### Required Files

Place these files in the same directory:
1. `ETPL.py` - Core ETPL compiler/parser/interpreter (provided)
2. `etpl_ide_stage1.py` - Base IDE framework
3. `etpl_ide_stage2_enhancements.py` - Advanced features
4. `etpl_ide_launcher.py` - Integrated launcher

### Launch the IDE

**Option 1: Full integrated version (Stages 1 + 2)**
```bash
python etpl_ide_launcher.py
```

**Option 2: Stage 1 only (simpler, fewer dependencies)**
```bash
python etpl_ide_stage1.py
```

---

## Feature Overview

### Stage 1 Features

#### File Management
- **New File** (Ctrl+N) - Create new .pdt file
- **Open File** (Ctrl+O) - Open existing .pdt or .py files
- **Save** (Ctrl+S) - Save current file
- **Save As** (Ctrl+Shift+S) - Save with new name
- **Multi-tab** - Work on multiple files simultaneously

#### ETPL Editing
- **Syntax Highlighting** - Full ET symbol recognition
  - Primitives (P, D, T) in magenta
  - Operators (∘, →, λ) in cyan
  - Math symbols in blue
  - Quantum symbols (ψ, ∞, Ω) in gold
  - Strings, numbers, comments color-coded

- **Symbol Shortcuts** - Quick insertion of ET symbols:
  - `Ctrl+O` → ∘ (binding operator)
  - `Ctrl+R` → → (arrow/path)
  - `Ctrl+L` → λ (lambda)
  - `Ctrl+I` → ∞ (infinity)
  - `Ctrl+P` → ψ (quantum psi)

- **Symbol Menu** - Edit → Insert ET Symbol submenu for all symbols

#### ETPL Tools
- **Parse** (F5) - Validate syntax and check ET completeness
- **Interpret** (F6) - Execute ETPL code
- **Compile** (F7) - Compile to executable binary
- **Compile Quantum** - Compile to QASM quantum circuit

#### Translation
- **Python → ETPL** - Convert Python files to ETPL
- **Binary → ETPL** - Reverse engineer executables to ETPL
- **Import Tracing** - Analyze full dependency chains

#### Console
- Color-coded output (red errors, green success, blue info)
- Clear operation feedback
- Execution results display

#### Self-Hosting
- **Build Self-Hosting Package** - Convert IDE itself to .pdt → .exe
- Creates standalone executable
- First step toward pure ETPL development

### Stage 2 Enhancements

#### Real-Time Error Detection
- **Syntax Errors** - Missing assignments, malformed structures
- **Semantic Errors** - Unbalanced brackets/parentheses
- **Undefined References** - Identifier usage before definition
- **Binding Consistency** - PDT completeness validation
- **Error Panel** - List of all errors/warnings
- **Click to Navigate** - Jump to error location in code

#### Code Intelligence
- **Auto-completion** - Keyword and symbol suggestions
- **Template Expansion** - Quick code templates:
  - P primitive template
  - D descriptor template  
  - T traverser template
  - Complete PDT binding
  - Loop structures
  - Exception handling
  - Quantum operations

#### Enhanced Import Analysis
- **Dependency Graph** - Visual representation of imports
- **Cycle Detection** - Identifies circular dependencies
- **Metrics**:
  - Total files count
  - Total dependencies
  - Maximum depth
  - Circular dependency warnings
- **Graph View** - Interactive visualization with color coding

---

## Usage Examples

### Example 1: Simple Point Definition

```etpl
# Create a Point with state
P myData = 42

# Create another Point
P result = 0
```

**Actions:**
1. Type the code
2. Press F5 to parse - validates syntax
3. Press F6 to interpret - executes code
4. Check console for results

### Example 2: Descriptor (Function)

```etpl
# Define a descriptor that increments a value
D increment = λ x . x + 1

# Use it with a point
P number = 10
P incremented = increment ∘ number
```

**Symbol insertion:**
- Type "D increment = " 
- Press Ctrl+L for λ
- Continue typing "x . x + 1"

### Example 3: Complete PDT Binding

```etpl
# Point substrate
P data = 100

# Descriptor constraint
D validate = λ x . x > 0

# Traverser agency
T processor = → data

# Complete binding (Eq 186)
P result = data ∘ validate ∘ processor
```

**Using shortcuts:**
- Ctrl+O for each ∘ operator
- Ctrl+R for → arrow

### Example 4: Infinite Loop with Bound

```etpl
# Infinite loop with finite bound
T counter = ∞ (i < 10) (D i)
```

**Symbol insertion:**
- Ctrl+I for ∞ symbol

### Example 5: Exception Handling

```etpl
# Exception ground with handler
T safe_divide = → a / b → E "Division error"
```

**Arrow insertion:**
- Ctrl+R for each → operator

### Example 6: Quantum Wavefunction

```etpl
# Quantum wave definition
ψ n l m . hydrogen_state
```

**Symbol insertion:**
- Ctrl+P for ψ symbol

### Example 7: Manifold Literal

```etpl
# Define a manifold
P elements = manifold [1, 2, 3, 4, 5]
```

---

## Workflow Examples

### Workflow 1: Parse → Interpret → Compile

```
1. Write ETPL code in editor
2. F5 - Parse to check syntax
3. Review any errors in Error Panel
4. Fix errors, repeat F5 until clean
5. F6 - Interpret to test execution
6. Check console output
7. F7 - Compile to create .exe
8. Output file created alongside .pdt
```

### Workflow 2: Python Translation

```
1. File → Open - select .py file
2. Tools → Translate Python to ETPL
3. New tab opens with translated code
4. Review and refine translation
5. Save as .pdt file
6. F7 to compile
```

### Workflow 3: Import Analysis

```
1. Open your main .pdt file
2. Tools → Trace Import Chain
3. View import tree in panel
4. Check Import Graph view for visualization
5. Look for circular dependencies (red dashed lines)
6. Review metrics in console
```

### Workflow 4: Self-Hosting Build

```
1. Ensure ETPL.py and IDE scripts are available
2. Tools → Build Self-Hosting Package
3. Select output directory
4. Wait for translation and compilation
5. Result: etpl_ide_unified.exe
6. Can now distribute standalone IDE!
```

---

## Advanced Features

### Error Detection Details

The IDE detects:

**Syntax Errors:**
- P/D/T without assignment
- Lambda without parameters
- Infinite loop without condition/bound
- Exception handler without expression
- Quantum wave without parameters
- Orphaned binding operators
- Manifold without brackets

**Semantic Errors:**
- Unbalanced `[]` brackets
- Unbalanced `()` parentheses
- Undefined identifier references
- Incomplete PDT bindings

### Import Tracing Details

The enhanced tracer:
- Recursively follows all imports
- Detects cycles (A imports B, B imports A)
- Calculates dependency depth
- Counts total files and edges
- Visualizes as interactive graph

**Graph Color Coding:**
- Magenta node = Root file
- Blue nodes = Dependencies
- Red nodes = Part of circular dependency
- Red dashed lines = Circular edges

### Code Completion

Trigger completion:
- Start typing keyword (P, D, T, etc.)
- Press Ctrl+Space (in future versions)
- Backslash for symbol insertion

Templates include:
- `P` expands to `P {id} = {value}`
- `D` expands to `D {id} = λ {params} . {body}`
- `T` expands to `T {id} = → {path}`
- `loop` expands to `∞ ({condition}) (D {bound})`

---

## Troubleshooting

### IDE Won't Start

**Issue:** PyQt5 not found
**Solution:**
```bash
pip install PyQt5 --break-system-packages
```

**Issue:** ETPL.py not found
**Solution:** Ensure ETPL.py is in same directory as IDE scripts

### Compilation Errors

**Issue:** "LLVM not found"
**Solution:**
```bash
pip install llvmlite --break-system-packages
```

**Issue:** "Capstone not found" (binary translation)
**Solution:**
```bash
pip install capstone --break-system-packages
```

### Parsing Errors

**Issue:** "Unknown symbol"
**Solution:** Use symbol shortcuts or Insert Symbol menu

**Issue:** "Parse gap" error
**Solution:** Check for incomplete bindings, missing operators

### Self-Hosting Build Fails

**Issue:** Import errors during translation
**Solution:** Ensure all ET library imports are available

**Issue:** Compilation timeout
**Solution:** Large files take time; wait for completion

---

## Tips & Best Practices

### Writing ETPL Code

1. **Start Simple** - Begin with basic P/D/T definitions
2. **Use Shortcuts** - Ctrl+O/R/L/I/P for symbols
3. **Parse Often** - F5 after each major change
4. **Check Errors** - Watch Error Panel for real-time feedback
5. **Comment Code** - Use `#` for documentation

### Using the IDE

1. **Multiple Files** - Use tabs for related files
2. **Save Frequently** - Ctrl+S after changes
3. **Import Tracing** - Use before compilation to see dependencies
4. **Console Output** - Monitor for execution feedback
5. **Error Navigation** - Click errors to jump to location

### Performance

1. **Large Files** - May take time to parse/compile
2. **Import Chains** - Deep chains increase analysis time
3. **Binary Translation** - Very slow for large executables
4. **Graph View** - Many nodes may need zoom/pan

---

## Keyboard Reference Card

```
FILE OPERATIONS
Ctrl+N          New file
Ctrl+O          Open file
Ctrl+S          Save
Ctrl+Shift+S    Save As
Ctrl+Q          Quit

EDITING
Ctrl+Z          Undo
Ctrl+Y          Redo
Ctrl+X          Cut
Ctrl+C          Copy
Ctrl+V          Paste

ET SYMBOLS
Ctrl+O          ∘ (binding)
Ctrl+R          → (arrow)
Ctrl+L          λ (lambda)
Ctrl+I          ∞ (infinity)
Ctrl+P          ψ (quantum)

RUN OPERATIONS
F5              Parse
F6              Interpret
F7              Compile to binary
```

---

## Next Steps

### Immediate (You can do now)
1. Launch IDE and explore interface
2. Create simple .pdt files
3. Test parse, interpret, compile
4. Try Python translation
5. Build self-hosting package

### Short Term (Stage 3 development)
- Advanced code intelligence
- Refactoring tools
- Debugging support
- Enhanced build system

### Long Term (Stages 4-5)
- Version control integration
- Testing framework
- Package management
- Pure ETPL self-hosting

---

## Support & Resources

**Documentation:**
- `ETPL_IDE_ROADMAP.md` - Complete development plan
- `ETPL_Language_Reference_Manual_v0_1.md` - ETPL language spec
- Exception Theory documentation - Core theory

**Files in Package:**
- `etpl_ide_stage1.py` - Base IDE
- `etpl_ide_stage2_enhancements.py` - Advanced features
- `etpl_ide_launcher.py` - Integrated launcher
- `SETUP_AND_USAGE_GUIDE.md` - This file

**Getting Help:**
- Check console output for error messages
- Review Error Panel for syntax issues
- Consult ET equations for mathematical derivations
- Test with ET Scanner for validation

---

## License

Created by Aevum Defluo
Based on Exception Theory
All components derived from ET primitives (P, D, T)

"For every exception there is an exception, except the exception"

© 2025 - All implementations derive from ET foundational mathematics

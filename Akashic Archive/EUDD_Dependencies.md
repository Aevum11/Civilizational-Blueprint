# EUDD — Required Dependencies

**All libraries must be available as static libraries for MSVC (Visual Studio 2022 Build Tools).**
**Recommended installation: vcpkg with `--triplet=x64-windows-static`**

## Core Arithmetic (Required — Stage 1)

| Library | Version | vcpkg Command | Purpose |
|---|---|---|---|
| **GMP** | ≥ 6.3 | `vcpkg install gmp:x64-windows-static` | Arbitrary-precision integer/rational arithmetic |
| **MPFR** | ≥ 4.2 | `vcpkg install mpfr:x64-windows-static` | 400-bit (120-dps) floating-point — the ET precision substrate |
| **FLINT** | ≥ 3.0 | `vcpkg install flint:x64-windows-static` | Special functions: ζ, Γ, polylog, hypergeometric via integrated Arb |

## GUI & Rendering (Required — Later Stages)

| Library | Version | vcpkg Command | Purpose |
|---|---|---|---|
| **GLFW** | ≥ 3.3 | `vcpkg install glfw3:x64-windows-static` | Window management, OpenGL context, input |
| **Dear ImGui** | ≥ 1.90 | `vcpkg install imgui[opengl3-binding,glfw-binding]:x64-windows-static` | Immediate-mode GUI widgets |
| **ImPlot** | ≥ 0.16 | `vcpkg install implot:x64-windows-static` | Charts, time-series, data visualization |

## Data Interchange (Required — Later Stages)

| Library | Version | vcpkg Command | Purpose |
|---|---|---|---|
| **yyjson** | ≥ 0.8 | `vcpkg install yyjson:x64-windows-static` | JSON protocol for API and extensions |

## Build Tools

| Tool | Version | Purpose |
|---|---|---|
| **CMake** | ≥ 3.24 | Build system |
| **MSVC** | VS 2022 Build Tools (v143) | C++17/20 compiler |
| **vcpkg** | Latest | Package management (set `VCPKG_ROOT` and pass `-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake`) |

## vcpkg Setup (one-time)

```batch
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
bootstrap-vcpkg.bat
set VCPKG_ROOT=C:\vcpkg
set VCPKG_DEFAULT_TRIPLET=x64-windows-static

vcpkg install gmp mpfr flint
```

## CMake Configuration

```batch
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows-static
cmake --build build --config Release
```

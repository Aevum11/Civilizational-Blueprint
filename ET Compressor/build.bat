@echo off
REM ═══════════════════════════════════════════════════════════════════════
REM Exception Theory — CDF Compressor Build Script
REM Double-click to build: et_pattern_engine.dll + et_cdf_compressor.exe
REM P ∘ D ∘ T = E
REM ═══════════════════════════════════════════════════════════════════════

cd /d "%~dp0"

echo ═══════════════════════════════════════════════════════════════
echo  ET CDF Compressor — Build Script
echo ═══════════════════════════════════════════════════════════════
echo.

REM ── Find Python ─────────────────────────────────────────────────
REM Try each command by actually running it. Do NOT use "where"
REM because Windows Store aliases and PATH issues make it unreliable.

set "PY="
py --version >nul 2>&1 && (set "PY=py" & goto :found_python)
python --version >nul 2>&1 && (set "PY=python" & goto :found_python)
python3 --version >nul 2>&1 && (set "PY=python3" & goto :found_python)

echo ERROR: Python not found.
echo        Tried: py, python, python3
echo        Install Python from https://www.python.org/downloads/
echo        and check "Add Python to PATH" during install.
pause
exit /b 1

:found_python
echo [INFO] Python command: %PY%
%PY% --version
echo.

REM ── Step 1: Compile et_pattern_engine.dll ──────────────────────

echo [1/3] Compiling C pattern engine...

set "DLL_NAME=et_pattern_engine.dll"
set "C_SRC=et_pattern_engine.c"
set "DLL_BUILT=0"

if not exist "%C_SRC%" (
    echo     %C_SRC% not found — .exe will auto-compile from external .c file at runtime.
    goto :skip_dll
)

REM Try cl.exe directly (works if running from Developer Command Prompt)
cl /nologo /O2 /LD /Fe:"%DLL_NAME%" "%C_SRC%" >nul 2>&1 && (
    echo     OK: %DLL_NAME% compiled with cl.exe
    set "DLL_BUILT=1"
    goto :dll_done
)

REM Try finding vcvarsall.bat for VS2022
for %%V in (
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
) do (
    if exist %%V (
        echo     Setting up MSVC via: %%V
        call %%V x64 >nul 2>&1
        cl /nologo /O2 /LD /Fe:"%DLL_NAME%" "%C_SRC%" >nul 2>&1 && (
            echo     OK: %DLL_NAME% compiled with MSVC
            set "DLL_BUILT=1"
            goto :dll_done
        )
    )
)

REM Try gcc
gcc --version >nul 2>&1 && (
    gcc -shared -O2 -o "%DLL_NAME%" "%C_SRC%" -lm >nul 2>&1 && (
        echo     OK: %DLL_NAME% compiled with gcc
        set "DLL_BUILT=1"
        goto :dll_done
    )
)

echo     No C compiler worked — ensure .c file is alongside .exe for runtime auto-compilation.

:dll_done
:skip_dll

REM Clean MSVC leftovers
del /q et_pattern_engine.obj et_pattern_engine.lib et_pattern_engine.exp 2>nul
echo.

REM ── Step 2: Verify PyInstaller ─────────────────────────────────

echo [2/3] Checking PyInstaller...

%PY% -c "import PyInstaller; print('    PyInstaller ' + PyInstaller.__version__ + ' found')" 2>nul && goto :has_pyinstaller

echo     PyInstaller not found — installing...
%PY% -m pip install pyinstaller
%PY% -c "import PyInstaller" 2>nul && goto :has_pyinstaller

echo     ERROR: Cannot install PyInstaller.
echo     Run manually: %PY% -m pip install pyinstaller
pause
exit /b 1

:has_pyinstaller
echo.

REM ── Step 3: Build single .exe ──────────────────────────────────

echo [3/3] Building single .exe...

set "PY_SCRIPT=et_cdf_compressor.py"

if not exist "%PY_SCRIPT%" (
    echo     ERROR: %PY_SCRIPT% not found in %cd%
    pause
    exit /b 1
)

REM Clean old PyInstaller build artifacts (NOT the spec — it has proper imports
REM and dynamic DLL detection. Deleting it would destroy the ET-derived fixes.)
if exist "build" rmdir /s /q "build" >nul 2>&1
if exist "dist" rmdir /s /q "dist" >nul 2>&1

REM ── Build Strategy (Descriptor Gap Principle): ──
REM The spec file detects the DLL dynamically via os.path.isfile at build time.
REM Subsumption Law: one spec file subsumes both DLL-present and DLL-absent
REM configurations without remainder — no conditional branching needed.
if exist "et_cdf_compressor.spec" (
    echo     Using spec file ^(DLL detection is dynamic^)...
    if "%DLL_BUILT%"=="1" echo     DLL detected: %DLL_NAME% will be bundled.
    if not "%DLL_BUILT%"=="1" echo     No DLL: .exe will auto-compile from external .c file at runtime.
    %PY% -m PyInstaller et_cdf_compressor.spec --noconfirm --clean
) else (
    echo     No spec file found — generating build from command line...
    if "%DLL_BUILT%"=="1" (
        echo     Bundling %DLL_NAME% into .exe...
        %PY% -m PyInstaller --onefile --windowed --clean --add-binary "%DLL_NAME%;." --name "et_cdf_compressor" "%PY_SCRIPT%" --noconfirm
    ) else (
        echo     Building .exe without DLL — ensure .c file is alongside .exe for runtime auto-compilation.
        %PY% -m PyInstaller --onefile --windowed --clean --name "et_cdf_compressor" "%PY_SCRIPT%" --noconfirm
    )
)

if not exist "dist\et_cdf_compressor.exe" (
    echo.
    echo     ERROR: Build failed. Check output above.
    pause
    exit /b 1
)

echo.
echo ═══════════════════════════════════════════════════════════════
echo  BUILD COMPLETE
for %%F in ("dist\et_cdf_compressor.exe") do echo  Output: %%~fF  (%%~zF bytes)
echo ═══════════════════════════════════════════════════════════════
echo.
pause
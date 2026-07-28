@echo off
setlocal enabledelayedexpansion

set LOGFILE=%~dp0vcpkg_build_log.txt
set SCRIPTDIR=%~dp0

echo EUDD Dependency Fix — %date% %time% > "%LOGFILE%"

echo ===================================================
echo  EUDD Dependency Fix — GMP VPATH Patch
echo  Log: %LOGFILE%
echo ===================================================
echo.

cd /d C:\vcpkg
if not exist "C:\vcpkg\vcpkg.exe" (
    echo [ERROR] vcpkg.exe not found at C:\vcpkg
    pause
    exit /b 1
)

:: ── Step 1: Clean ────────────────────────────────────
echo [1/5] Cleaning previous builds...
echo [1/5] Cleaning >> "%LOGFILE%"
if exist "C:\vcpkg\buildtrees\gmp" rd /s /q "C:\vcpkg\buildtrees\gmp"
if exist "C:\vcpkg\buildtrees\mpfr" rd /s /q "C:\vcpkg\buildtrees\mpfr"
if exist "C:\vcpkg\buildtrees\flint" rd /s /q "C:\vcpkg\buildtrees\flint"
.\vcpkg remove gmp:x64-windows-static --recurse 2>nul
.\vcpkg remove mpfr:x64-windows-static --recurse 2>nul
.\vcpkg remove flint:x64-windows-static --recurse 2>nul
echo   Done.

:: ── Step 2: Patch GMP portfile ───────────────────────
echo.
echo [2/5] Creating patched GMP overlay port...
echo [2/5] Patching portfile >> "%LOGFILE%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPTDIR%patch_gmp_port.ps1"
if errorlevel 1 (
    echo   [ERROR] Patch script failed
    echo   [ERROR] Patch script failed >> "%LOGFILE%"
    pause
    exit /b 1
)

:: Verify the overlay was created
if not exist "C:\vcpkg-overlay\gmp\portfile.cmake" (
    echo   [ERROR] Overlay portfile not found after patching
    pause
    exit /b 1
)
echo   Overlay port ready at C:\vcpkg-overlay\gmp

:: ── Step 3: Build GMP ────────────────────────────────
echo.
echo [3/5] Building GMP with VPATH fix (5-15 min, do NOT close)...
echo [3/5] Building GMP — start %time% >> "%LOGFILE%"
set VCPKG_MAX_CONCURRENCY=1
.\vcpkg install gmp:x64-windows-static --overlay-ports=C:\vcpkg-overlay --no-binarycaching
set GMP_EXIT=%errorlevel%
echo GMP exit code: %GMP_EXIT% >> "%LOGFILE%"
echo GMP end: %time% >> "%LOGFILE%"

if %GMP_EXIT% neq 0 (
    echo   [ERROR] GMP build failed with exit code %GMP_EXIT%
    echo   [ERROR] GMP failed >> "%LOGFILE%"
    echo   Dumping error logs...
    for /r "C:\vcpkg\buildtrees\gmp" %%f in (*err.log) do (
        echo === %%~nxf === >> "%LOGFILE%"
        type "%%f" >> "%LOGFILE%"
        echo.
        echo --- %%~nxf ---
        type "%%f"
    )
    echo. >> "%LOGFILE%"
    echo === build-x64-windows-dbg-out.log (last 40 lines) === >> "%LOGFILE%"
    if exist "C:\vcpkg\buildtrees\gmp\build-x64-windows-dbg-out.log" (
        powershell -Command "Get-Content 'C:\vcpkg\buildtrees\gmp\build-x64-windows-dbg-out.log' -Tail 40" >> "%LOGFILE%"
    )
    echo.
    echo   Upload %LOGFILE% to Claude.
    pause
    exit /b 1
)
echo   GMP built successfully!

:: ── Step 4: Build MPFR ───────────────────────────────
echo.
echo [4/5] Building MPFR (5-10 min)...
echo [4/5] Building MPFR — start %time% >> "%LOGFILE%"
.\vcpkg install mpfr:x64-windows-static --no-binarycaching
set MPFR_EXIT=%errorlevel%
echo MPFR exit code: %MPFR_EXIT% >> "%LOGFILE%"

if %MPFR_EXIT% neq 0 (
    echo   [ERROR] MPFR build failed
    for /r "C:\vcpkg\buildtrees\mpfr" %%f in (*err.log) do (
        echo === %%~nxf === >> "%LOGFILE%"
        type "%%f" >> "%LOGFILE%"
        type "%%f"
    )
    echo   Upload %LOGFILE% to Claude.
    pause
    exit /b 1
)
echo   MPFR built successfully!

:: ── Step 5: Build FLINT ──────────────────────────────
echo.
echo [5/5] Building FLINT (5-15 min)...
echo [5/5] Building FLINT — start %time% >> "%LOGFILE%"
.\vcpkg install flint:x64-windows-static --no-binarycaching
set FLINT_EXIT=%errorlevel%
echo FLINT exit code: %FLINT_EXIT% >> "%LOGFILE%"

if %FLINT_EXIT% neq 0 (
    echo   [ERROR] FLINT build failed
    for /r "C:\vcpkg\buildtrees\flint" %%f in (*err.log) do (
        echo === %%~nxf === >> "%LOGFILE%"
        type "%%f" >> "%LOGFILE%"
        type "%%f"
    )
    echo   Upload %LOGFILE% to Claude.
    pause
    exit /b 1
)
echo   FLINT built successfully!

:: ── Verify ───────────────────────────────────────────
echo.
echo ===================================================
echo  Verification
echo ===================================================
echo === VERIFICATION === >> "%LOGFILE%"

set ID=C:\vcpkg\installed\x64-windows-static

if exist "%ID%\include\gmp.h" (echo   [OK] gmp.h) else (echo   [FAIL] gmp.h MISSING)
if exist "%ID%\include\mpfr.h" (echo   [OK] mpfr.h) else (echo   [FAIL] mpfr.h MISSING)
if exist "%ID%\include\flint\flint.h" (echo   [OK] flint/flint.h) else (echo   [FAIL] flint/flint.h MISSING)

if exist "%ID%\include\gmp.h" (echo   [OK] gmp.h >> "%LOGFILE%") else (echo   [FAIL] gmp.h >> "%LOGFILE%")
if exist "%ID%\include\mpfr.h" (echo   [OK] mpfr.h >> "%LOGFILE%") else (echo   [FAIL] mpfr.h >> "%LOGFILE%")
if exist "%ID%\include\flint\flint.h" (echo   [OK] flint/flint.h >> "%LOGFILE%") else (echo   [FAIL] flint/flint.h >> "%LOGFILE%")

echo.
echo --- Libraries ---
dir "%ID%\lib\*gmp*" 2>nul
dir "%ID%\lib\*mpfr*" 2>nul
dir "%ID%\lib\*flint*" 2>nul

echo.
echo ===================================================
echo  FINISHED — %date% %time%
echo  Log: %LOGFILE%
echo.
echo  In CLion, set CMake options to:
echo    -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
echo    -DVCPKG_TARGET_TRIPLET=x64-windows-static
echo.
echo  If [FAIL], upload %LOGFILE% to Claude.
echo ===================================================
pause

@echo off
setlocal EnableDelayedExpansion
::
:: build.bat
:: ET32 Bridge — Complete Build Script
::
:: Derived from P o D o T = E.
::
:: Build sequence (T traverses D to produce E):
::   Phase 1 — Compile et_bridge32.dll  (32-bit C DLL)
::   Phase 2 — Build ET32_Bridge.exe    (64-bit Python broker, PyInstaller)
::   Phase 3 — Build ET32_Bridge_Helper.exe (32-bit Python helper, PyInstaller)
::   Phase 4 — Assemble distribution folder
::
:: ET constants embedded in the build:
::   S  = 12    (12 Python modules, 12 retry count in DLL)
::   K  = 2/3   (Koide ratio — used as optimisation threshold)
::   hd = 4096  (digital action quantum — DLL buffer unit)
::
:: Prerequisites:
::   64-bit Python 3.9+  (in PATH or set PYTHON64)
::   32-bit Python 3.9+  (set PYTHON32 or skip helper build)
::   MinGW-w64 or MSVC   (for DLL compilation; set COMPILER=msvc or COMPILER=mingw)
::   pyinstaller         (pip install pyinstaller pystray pillow pywin32)
::
:: Usage:
::   build.bat                        -- auto-detect compiler, full build
::   build.bat dll                    -- compile DLL only
::   build.bat broker                 -- build broker exe only
::   build.bat helper                 -- build helper exe only
::   build.bat dist                   -- assemble distribution only
::   build.bat clean                  -- remove build artefacts
::
:: SPDX-License-Identifier: MIT
:: Copyright (c) 2026 Michael James Muller (Aevum Defluo) — Exception Theory
::

:: ---------------------------------------------------------------------------
:: Configuration — override these via environment variables if needed
:: ---------------------------------------------------------------------------

:: Path to 64-bit Python executable
if not defined PYTHON64 (
    where python >nul 2>&1 && set PYTHON64=python
    if not defined PYTHON64 set PYTHON64=python
)

:: Path to 32-bit Python executable (needed only for helper build)
if not defined PYTHON32 (
    if exist "C:\Python39-32\python.exe"  set PYTHON32=C:\Python39-32\python.exe
    if exist "C:\Python310-32\python.exe" set PYTHON32=C:\Python310-32\python.exe
    if exist "C:\Python311-32\python.exe" set PYTHON32=C:\Python311-32\python.exe
    if exist "C:\Python312-32\python.exe" set PYTHON32=C:\Python312-32\python.exe
)

:: Compiler preference: "mingw" or "msvc" (auto-detected if not set)
if not defined COMPILER (
    where gcc >nul 2>&1 && set COMPILER=mingw
    if not defined COMPILER (
        where cl >nul 2>&1 && set COMPILER=msvc
    )
    if not defined COMPILER (
        echo [ET32] WARNING: No C compiler found. DLL build will be skipped.
        echo [ET32]          Install MinGW-w64 or MSVC and ensure it is in PATH.
        set COMPILER=none
    )
)

:: Distribution output directory
if not defined DIST_DIR set DIST_DIR=%~dp0dist\ET32_Bridge_Release

:: ET build constants (S=12)
set ET_S=12
set SCRIPT_DIR=%~dp0

:: ---------------------------------------------------------------------------
:: Argument parsing
:: ---------------------------------------------------------------------------
set BUILD_DLL=0
set BUILD_BROKER=0
set BUILD_HELPER=0
set BUILD_DIST=0
set BUILD_CLEAN=0

if "%~1"==""       goto :build_all
if "%~1"=="dll"    ( set BUILD_DLL=1     & goto :run )
if "%~1"=="broker" ( set BUILD_BROKER=1  & goto :run )
if "%~1"=="helper" ( set BUILD_HELPER=1  & goto :run )
if "%~1"=="dist"   ( set BUILD_DIST=1    & goto :run )
if "%~1"=="clean"  ( set BUILD_CLEAN=1   & goto :run )
echo [ET32] Unknown argument: %~1
echo [ET32] Usage: build.bat [dll^|broker^|helper^|dist^|clean]
exit /b 1

:build_all
set BUILD_DLL=1
set BUILD_BROKER=1
set BUILD_HELPER=1
set BUILD_DIST=1
goto :run

:: ---------------------------------------------------------------------------
:run
:: ---------------------------------------------------------------------------
echo.
echo ============================================================
echo  ET32 Bridge Build System  ^|  P o D o T = E
echo  S=%ET_S%  K=2/3  hd=4096  IPC_BUFFER=49152
echo ============================================================
echo.

cd /d "%SCRIPT_DIR%"

:: ---------------------------------------------------------------------------
:: CLEAN
:: ---------------------------------------------------------------------------
if %BUILD_CLEAN%==1 (
    echo [ET32] Cleaning build artefacts...
    if exist build       rmdir /s /q build
    if exist dist        rmdir /s /q dist
    if exist __pycache__ rmdir /s /q __pycache__
    if exist et_bridge\__pycache__ rmdir /s /q et_bridge\__pycache__
    if exist et_bridge32.dll       del /f et_bridge32.dll
    if exist et_bridge32.obj       del /f et_bridge32.obj
    if exist et_bridge32.exp       del /f et_bridge32.exp
    if exist et_bridge32.lib       del /f et_bridge32.lib
    if exist et_bridge32.def       del /f et_bridge32.def
    echo [ET32] Clean complete.
    goto :eof
)

:: ---------------------------------------------------------------------------
:: PHASE 1 — Compile et_bridge32.dll
:: ---------------------------------------------------------------------------
if %BUILD_DLL%==1 (
    echo [ET32] Phase 1: Compiling et_bridge32.dll ^(32-bit^)...

    if not exist et_bridge32.c (
        echo [ET32] ERROR: et_bridge32.c not found.
        exit /b 1
    )

    if "%COMPILER%"=="mingw" (
        echo [ET32]   Compiler: MinGW-w64 gcc ^(32-bit target^)

        :: Write a minimal .def file for explicit exports
        (
            echo LIBRARY et_bridge32
            echo EXPORTS
            echo     ET32_Init
            echo     ET32_Shutdown
            echo     ET32_IsConnected
            echo     ET32_GetVersion
            echo     ET32_BridgeVirtualAlloc
            echo     ET32_BridgeVirtualAlloc64
            echo     ET32_BridgeLoadLibrary64
            echo     ET32_BridgeGetProcAddress64
            echo     ET32_BridgeCall64
            echo     ET32_BridgeRegOpenKey64
            echo     ET32_PythonExec64
            echo     ET32_UniversalHook
            echo     ET32_AWE_ReserveWindow
            echo     ET32_AWE_MapPages
            echo     ET32_AWE_UnmapWindow
            echo     ET32_AWE_SlideWindow
            echo     ET32_AWE_ReleaseWindow
            echo     ET32_AWE_DirectAlloc
        ) > et_bridge32.def

        gcc -m32 -O2 -shared ^
            -o et_bridge32.dll ^
            et_bridge32.c ^
            et_bridge32.def ^
            -lkernel32 -ladvapi32 ^
            -Wl,--subsystem,windows ^
            -Wl,--enable-stdcall-fixup ^
            -Wall -Wextra -Wno-unused-parameter

        if errorlevel 1 (
            echo [ET32] ERROR: MinGW DLL compilation failed.
            exit /b 1
        )
        echo [ET32]   et_bridge32.dll compiled successfully ^(MinGW^).

    ) else if "%COMPILER%"=="msvc" (
        echo [ET32]   Compiler: MSVC cl ^(x86 target^)

        :: Write .def file
        (
            echo LIBRARY et_bridge32
            echo EXPORTS
            echo     ET32_Init
            echo     ET32_Shutdown
            echo     ET32_IsConnected
            echo     ET32_GetVersion
            echo     ET32_BridgeVirtualAlloc
            echo     ET32_BridgeVirtualAlloc64
            echo     ET32_BridgeLoadLibrary64
            echo     ET32_BridgeGetProcAddress64
            echo     ET32_BridgeCall64
            echo     ET32_BridgeRegOpenKey64
            echo     ET32_PythonExec64
            echo     ET32_UniversalHook
            echo     ET32_AWE_ReserveWindow
            echo     ET32_AWE_MapPages
            echo     ET32_AWE_UnmapWindow
            echo     ET32_AWE_SlideWindow
            echo     ET32_AWE_ReleaseWindow
            echo     ET32_AWE_DirectAlloc
        ) > et_bridge32.def

        :: Detect and activate x86 MSVC environment if not already set
        if not defined VCINSTALLDIR (
            for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath 2^>nul`) do (
                call "%%i\VC\Auxiliary\Build\vcvars32.bat" >nul 2>&1
            )
        )

        cl /LD /Ox /arch:IA32 ^
           /Fe:et_bridge32.dll ^
           et_bridge32.c ^
           /link et_bridge32.def ^
           kernel32.lib advapi32.lib

        if errorlevel 1 (
            echo [ET32] ERROR: MSVC DLL compilation failed.
            exit /b 1
        )
        echo [ET32]   et_bridge32.dll compiled successfully ^(MSVC^).

    ) else (
        echo [ET32] WARNING: No C compiler available. Skipping DLL compilation.
        echo [ET32]          Place a pre-built et_bridge32.dll in this directory.
    )

    :: Verify DLL was produced
    if not exist et_bridge32.dll (
        echo [ET32] ERROR: et_bridge32.dll was not produced.
        exit /b 1
    )

    echo [ET32] Phase 1 complete.
    echo.
)

:: ---------------------------------------------------------------------------
:: PHASE 2 — Build 64-bit broker (ET32_Bridge.exe)
:: ---------------------------------------------------------------------------
if %BUILD_BROKER%==1 (
    echo [ET32] Phase 2: Building ET32_Bridge.exe ^(64-bit broker^)...

    :: Verify 64-bit Python
    %PYTHON64% -c "import sys; assert sys.maxsize > 2**32, 'Need 64-bit Python'" >nul 2>&1
    if errorlevel 1 (
        echo [ET32] ERROR: PYTHON64=%PYTHON64% is not a 64-bit Python interpreter.
        echo [ET32]        Set PYTHON64 to the path of a 64-bit Python 3.9+ executable.
        exit /b 1
    )

    :: Install/verify pyinstaller
    %PYTHON64% -m pyinstaller --version >nul 2>&1
    if errorlevel 1 (
        echo [ET32]   PyInstaller not found. Installing...
        %PYTHON64% -m pip install pyinstaller pystray pillow pywin32 --quiet
        if errorlevel 1 (
            echo [ET32] ERROR: Failed to install PyInstaller.
            exit /b 1
        )
    )

    :: Run PyInstaller for the broker
    %PYTHON64% -m pyinstaller ^
        --clean ^
        --noconfirm ^
        et32_bridge.spec

    if errorlevel 1 (
        echo [ET32] ERROR: Broker PyInstaller build failed.
        exit /b 1
    )

    echo [ET32] Phase 2 complete.
    echo.
)

:: ---------------------------------------------------------------------------
:: PHASE 3 — Build 32-bit helper (ET32_Bridge_Helper.exe)
:: ---------------------------------------------------------------------------
if %BUILD_HELPER%==1 (
    echo [ET32] Phase 3: Building ET32_Bridge_Helper.exe ^(32-bit helper^)...

    if not defined PYTHON32 (
        echo [ET32] WARNING: PYTHON32 not set. Skipping helper build.
        echo [ET32]          Set PYTHON32=^<path to 32-bit python.exe^> to enable this.
        goto :phase3_done
    )

    :: Verify 32-bit Python
    %PYTHON32% -c "import sys; assert sys.maxsize <= 2**32, 'Need 32-bit Python'" >nul 2>&1
    if errorlevel 1 (
        echo [ET32] WARNING: PYTHON32=%PYTHON32% is not a 32-bit Python interpreter.
        echo [ET32]          Skipping helper build.
        goto :phase3_done
    )

    %PYTHON32% -m pyinstaller --version >nul 2>&1
    if errorlevel 1 (
        echo [ET32]   PyInstaller not found in 32-bit Python. Installing...
        %PYTHON32% -m pip install pyinstaller pywin32 --quiet
        if errorlevel 1 (
            echo [ET32] ERROR: Failed to install PyInstaller into 32-bit Python.
            exit /b 1
        )
    )

    %PYTHON32% -m pyinstaller ^
        --clean ^
        --noconfirm ^
        et32_bridge_helper.spec

    if errorlevel 1 (
        echo [ET32] ERROR: Helper PyInstaller build failed.
        exit /b 1
    )

    echo [ET32] Phase 3 complete.

    :phase3_done
    echo.
)

:: ---------------------------------------------------------------------------
:: PHASE 4 — Assemble distribution
:: ---------------------------------------------------------------------------
if %BUILD_DIST%==1 (
    echo [ET32] Phase 4: Assembling distribution in %DIST_DIR%...

    if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"
    mkdir "%DIST_DIR%"

    :: Copy broker
    if exist "dist\ET32_Bridge" (
        xcopy /e /i /q "dist\ET32_Bridge\*" "%DIST_DIR%\" >nul
        echo [ET32]   Broker files copied.
    ) else (
        echo [ET32] WARNING: dist\ET32_Bridge not found. Run broker build first.
    )

    :: Copy helper alongside broker
    if exist "dist\ET32_Bridge_Helper" (
        if not exist "%DIST_DIR%\helper" mkdir "%DIST_DIR%\helper"
        xcopy /e /i /q "dist\ET32_Bridge_Helper\*" "%DIST_DIR%\helper\" >nul
        echo [ET32]   Helper files copied to helper\.
    ) else (
        echo [ET32] WARNING: dist\ET32_Bridge_Helper not found. Skipping helper copy.
    )

    :: Copy et_bridge32.dll to distribution root
    if exist et_bridge32.dll (
        copy /y et_bridge32.dll "%DIST_DIR%\et_bridge32.dll" >nul
        echo [ET32]   et_bridge32.dll copied.
    )

    :: Copy config template (if not already copied by PyInstaller datas)
    if not exist "%DIST_DIR%\config_template.json" (
        if exist config_template.json (
            copy /y config_template.json "%DIST_DIR%\config_template.json" >nul
            echo [ET32]   config_template.json copied.
        )
    )

    :: Copy C source for reference
    if exist et_bridge32.c (
        if not exist "%DIST_DIR%\src" mkdir "%DIST_DIR%\src"
        copy /y et_bridge32.c "%DIST_DIR%\src\et_bridge32.c" >nul
        echo [ET32]   et_bridge32.c copied to src\.
    )

    :: Write a README for the release
    (
        echo ET32 Bridge — Release Distribution
        echo P o D o T = E
        echo.
        echo Files:
        echo   ET32_Bridge.exe          64-bit broker ^(run this as Administrator^)
        echo   et_bridge32.dll          32-bit injectable DLL
        echo   config_template.json     Configuration template
        echo   helper\                  32-bit companion helper
        echo   src\                     C source for et_bridge32.dll
        echo.
        echo Quick start:
        echo   1. Copy config_template.json to et32_bridge_config.json
        echo   2. Edit et32_bridge_config.json: set exe_name and enabled=true
        echo   3. Run ET32_Bridge.exe --config et32_bridge_config.json
        echo.
        echo Generate a default config:
        echo   ET32_Bridge.exe --generate-config
        echo.
        echo Exception Theory: P o D o T = E
        echo S=12  K=2/3  V=1/12  hd=4096
    ) > "%DIST_DIR%\README.txt"

    echo [ET32] Phase 4 complete.
    echo.

    :: Compute and display distribution size
    set DIST_SIZE=0
    for /r "%DIST_DIR%" %%f in (*) do (
        set /a DIST_SIZE+=1
    )
    echo [ET32] Distribution assembled: %DIST_DIR%
    echo [ET32] File count: !DIST_SIZE!
)

:: ---------------------------------------------------------------------------
:: Summary
:: ---------------------------------------------------------------------------
echo.
echo ============================================================
echo  ET32 Bridge Build Complete
echo  P o D o T = E  ^|  V(E) = 0
echo ============================================================
echo.

if exist et_bridge32.dll (
    echo   [OK] et_bridge32.dll
) else (
    echo   [--] et_bridge32.dll   ^(not built^)
)

if exist "dist\ET32_Bridge\ET32_Bridge.exe" (
    echo   [OK] dist\ET32_Bridge\ET32_Bridge.exe
) else (
    echo   [--] ET32_Bridge.exe   ^(not built^)
)

if exist "dist\ET32_Bridge_Helper\ET32_Bridge_Helper.exe" (
    echo   [OK] dist\ET32_Bridge_Helper\ET32_Bridge_Helper.exe
) else (
    echo   [--] ET32_Bridge_Helper.exe   ^(not built / PYTHON32 not set^)
)

if exist "%DIST_DIR%\ET32_Bridge.exe" (
    echo   [OK] %DIST_DIR%\  ^(release distribution^)
)

echo.
endlocal
exit /b 0

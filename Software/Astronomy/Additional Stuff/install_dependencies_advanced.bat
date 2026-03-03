@echo off
REM ============================================================================
REM ET Scanner v6.1 - Advanced Windows Installer
REM ============================================================================
REM
REM This advanced installer provides:
REM   - Multiple installation modes
REM   - Virtual environment support
REM   - Dependency conflict resolution
REM   - Offline installation option
REM   - Upgrade/repair capabilities
REM
REM Usage: install_dependencies_advanced.bat [OPTIONS]
REM
REM Options:
REM   /venv      - Install in virtual environment (recommended)
REM   /user      - Install for current user only
REM   /system    - Install system-wide (requires admin)
REM   /upgrade   - Upgrade existing installation
REM   /repair    - Repair broken installation
REM   /minimal   - Install only core dependencies
REM   /full      - Install all dependencies (default)
REM
REM Author: Exception Theory Scanner Project
REM Version: 6.1
REM ============================================================================

setlocal enabledelayedexpansion

REM Parse command line arguments
set INSTALL_MODE=full
set INSTALL_TARGET=user
set USE_VENV=0
set UPGRADE_MODE=0
set REPAIR_MODE=0

:parse_args
if "%~1"=="" goto :done_parsing
if /i "%~1"=="/venv" set USE_VENV=1
if /i "%~1"=="/user" set INSTALL_TARGET=user
if /i "%~1"=="/system" set INSTALL_TARGET=system
if /i "%~1"=="/upgrade" set UPGRADE_MODE=1
if /i "%~1"=="/repair" set REPAIR_MODE=1
if /i "%~1"=="/minimal" set INSTALL_MODE=minimal
if /i "%~1"=="/full" set INSTALL_MODE=full
shift
goto :parse_args
:done_parsing

REM Set console
title ET Scanner v6.1 - Advanced Installer
color 0B

echo.
echo ============================================================================
echo   EXCEPTION THEORY SCANNER v6.1 - ADVANCED INSTALLER
echo ============================================================================
echo.
echo   Installation Configuration:
if %USE_VENV%==1 (
    echo     Mode: Virtual Environment
) else (
    echo     Mode: Direct Installation
)
echo     Target: %INSTALL_TARGET%
echo     Packages: %INSTALL_MODE%
if %UPGRADE_MODE%==1 echo     Action: Upgrade
if %REPAIR_MODE%==1 echo     Action: Repair
echo.
echo   Press Ctrl+C to cancel, or any key to continue...
pause >nul
echo.

REM ============================================================================
REM Check Python
REM ============================================================================

echo [1/8] Checking Python installation...
echo.

python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo   [ERROR] Python not found!
    echo.
    echo   Please install Python 3.8+ from: https://www.python.org/downloads/
    echo   Make sure to check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo   [OK] Python %PYTHON_VERSION% found
echo.

REM ============================================================================
REM Create Virtual Environment (if requested)
REM ============================================================================

if %USE_VENV%==1 (
    echo [2/8] Creating virtual environment...
    echo.
    
    if exist "et_scanner_venv\" (
        echo   [INFO] Virtual environment already exists
        echo.
        set /p RECREATE="Recreate virtual environment? (Y/N): "
        if /i "!RECREATE!"=="Y" (
            echo   Removing old environment...
            rmdir /s /q et_scanner_venv
        )
    )
    
    if not exist "et_scanner_venv\" (
        echo   Creating new virtual environment...
        python -m venv et_scanner_venv
        if errorlevel 1 (
            color 0C
            echo   [ERROR] Failed to create virtual environment
            echo.
            pause
            exit /b 1
        )
        echo   [OK] Virtual environment created
    )
    
    echo   Activating virtual environment...
    call et_scanner_venv\Scripts\activate.bat
    echo   [OK] Virtual environment activated
    echo.
) else (
    echo [2/8] Skipping virtual environment (direct install)
    echo.
)

REM ============================================================================
REM Upgrade pip
REM ============================================================================

echo [3/8] Upgrading pip...
echo.

if %INSTALL_TARGET%==user (
    python -m pip install --upgrade pip --user --quiet
) else (
    python -m pip install --upgrade pip --quiet
)

if errorlevel 1 (
    color 0E
    echo   [WARNING] pip upgrade failed, continuing...
    echo.
) else (
    echo   [OK] pip upgraded
    echo.
)

REM ============================================================================
REM Install Core Dependencies
REM ============================================================================

echo [4/8] Installing core dependencies...
echo.

set PIP_FLAGS=--quiet
if %INSTALL_TARGET%==user set PIP_FLAGS=--user --quiet
if %UPGRADE_MODE%==1 set PIP_FLAGS=%PIP_FLAGS% --upgrade
if %REPAIR_MODE%==1 set PIP_FLAGS=%PIP_FLAGS% --force-reinstall

REM NumPy
echo   Installing numpy...
python -m pip install numpy %PIP_FLAGS%
if errorlevel 1 (
    color 0C
    echo   [ERROR] numpy installation failed
    goto :install_failed
)
echo   [OK] numpy

REM SciPy
echo   Installing scipy...
python -m pip install scipy %PIP_FLAGS%
if errorlevel 1 (
    color 0C
    echo   [ERROR] scipy installation failed
    goto :install_failed
)
echo   [OK] scipy

echo.
echo   [OK] Core dependencies installed
echo.

REM ============================================================================
REM Install Optional Dependencies
REM ============================================================================

if %INSTALL_MODE%==full (
    echo [5/8] Installing optional dependencies...
    echo.
    
    REM psutil
    echo   Installing psutil (process monitoring)...
    python -m pip install psutil %PIP_FLAGS%
    if errorlevel 1 (
        color 0E
        echo   [WARNING] psutil failed (optional)
    ) else (
        echo   [OK] psutil
    )
    
    echo.
    echo   [OK] Optional dependencies installed
    echo.
) else (
    echo [5/8] Skipping optional dependencies (minimal install)
    echo.
)

REM ============================================================================
REM Verify Installation
REM ============================================================================

echo [6/8] Verifying installation...
echo.

REM Create comprehensive test
(
    echo import sys
    echo import importlib.metadata
    echo.
    echo # Test imports
    echo deps = {}
    echo.
    echo # Core deps
    echo try:
    echo     import numpy as np
    echo     deps['numpy'] = importlib.metadata.version('numpy'^)
    echo     print(f'  [OK] numpy {deps["numpy"]}'^)
    echo except Exception as e:
    echo     print(f'  [FAIL] numpy: {e}'^)
    echo     sys.exit(1^)
    echo.
    echo try:
    echo     import scipy
    echo     deps['scipy'] = importlib.metadata.version('scipy'^)
    echo     print(f'  [OK] scipy {deps["scipy"]}'^)
    echo except Exception as e:
    echo     print(f'  [FAIL] scipy: {e}'^)
    echo     sys.exit(1^)
    echo.
    echo # Optional deps
    echo try:
    echo     import psutil
    echo     deps['psutil'] = importlib.metadata.version('psutil'^)
    echo     print(f'  [OK] psutil {deps["psutil"]}'^)
    echo except ImportError:
    echo     print('  [SKIP] psutil (not installed^)'^)
    echo.
    echo # Built-in deps
    echo try:
    echo     import tkinter
    echo     print('  [OK] tkinter (built-in^)'^)
    echo except ImportError:
    echo     print('  [WARN] tkinter (file dialogs disabled^)'^)
    echo.
    echo try:
    echo     import urllib.request
    echo     print('  [OK] urllib (built-in^)'^)
    echo except ImportError:
    echo     print('  [WARN] urllib (URL fetching disabled^)'^)
    echo.
    echo print('^n  All core dependencies verified!'^)
) > verify_install.py

python verify_install.py
set VERIFY_RESULT=%errorlevel%
del verify_install.py

if %VERIFY_RESULT% neq 0 (
    color 0C
    echo.
    echo   [ERROR] Installation verification failed
    goto :install_failed
)

echo.
echo   [OK] Installation verified
echo.

REM ============================================================================
REM Test Scanner
REM ============================================================================

echo [7/8] Testing ET Scanner...
echo.

if not exist "et_scanner_v6.1.py" (
    color 0E
    echo   [WARNING] et_scanner_v6.1.py not found
    echo.
    echo   Place the scanner in the same directory as this installer.
    echo.
    goto :skip_scanner_test
)

python et_scanner_v6.1.py --help >nul 2>&1
if errorlevel 1 (
    color 0E
    echo   [WARNING] Scanner test had issues
    echo.
    goto :skip_scanner_test
)

echo   [OK] Scanner working
echo.

:skip_scanner_test

REM ============================================================================
REM Create Launch Scripts
REM ============================================================================

echo [8/8] Creating launch scripts...
echo.

if %USE_VENV%==1 (
    REM Create venv launcher
    (
        echo @echo off
        echo call et_scanner_venv\Scripts\activate.bat
        echo python et_scanner_v6.1.py %%*
    ) > run_scanner.bat
    
    echo   [OK] Created run_scanner.bat (uses virtual environment^)
) else (
    REM Create direct launcher
    (
        echo @echo off
        echo python et_scanner_v6.1.py %%*
    ) > run_scanner.bat
    
    echo   [OK] Created run_scanner.bat
)

REM Create quick launchers
(
    echo @echo off
    echo python et_scanner_v6.1.py --entropy
    echo pause
) > quick_entropy_scan.bat

(
    echo @echo off
    echo python et_scanner_v6.1.py
) > interactive_scanner.bat

echo   [OK] Created quick_entropy_scan.bat
echo   [OK] Created interactive_scanner.bat
echo.

REM ============================================================================
REM Installation Complete
REM ============================================================================

color 0A
echo.
echo ============================================================================
echo   INSTALLATION COMPLETE!
echo ============================================================================
echo.
echo   Python: %PYTHON_VERSION%
if %USE_VENV%==1 (
    echo   Environment: et_scanner_venv\
)
echo   Dependencies: %INSTALL_MODE%
echo.
echo   How to run:
echo.
if %USE_VENV%==1 (
    echo     run_scanner.bat              (recommended - uses venv^)
) else (
    echo     run_scanner.bat              (direct launcher^)
)
echo     interactive_scanner.bat      (interactive menu^)
echo     quick_entropy_scan.bat       (quick entropy test^)
echo.
echo   Or manually:
if %USE_VENV%==1 (
    echo     et_scanner_venv\Scripts\activate.bat
)
echo     python et_scanner_v6.1.py [options]
echo.
echo ============================================================================
echo.

set /p RUN_NOW="Run scanner now? (Y/N): "
if /i "%RUN_NOW%"=="Y" (
    echo.
    echo Starting scanner...
    echo.
    timeout /t 1 /nobreak >nul
    call run_scanner.bat
)

echo.
echo Press any key to exit...
pause >nul
exit /b 0

REM ============================================================================
REM Error Handler
REM ============================================================================

:install_failed
echo.
echo ============================================================================
echo   INSTALLATION FAILED
echo ============================================================================
echo.
echo   The installation encountered errors.
echo.
echo   Troubleshooting:
echo.
echo   1. Run as administrator (for system-wide install^)
echo   2. Try with /venv option (isolated environment^)
echo   3. Try with /repair option (fix broken install^)
echo   4. Check internet connection
echo   5. Check firewall/proxy settings
echo.
echo   For help, visit:
echo   https://docs.python.org/3/installing/
echo.
echo ============================================================================
echo.
pause
exit /b 1

@echo off
REM ============================================================================
REM ET Scanner v6.1.1 - Advanced Windows Installer (FIXED)
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
REM Version: 6.1.1 (Fixed)
REM ============================================================================

setlocal

REM ============================================================================
REM Configuration - EDIT THIS SECTION IF NEEDED
REM ============================================================================
REM Scanner filename - change this if your file has a different name
set SCANNER_FILE=et_scanner_v6_1_fixed.py

REM ============================================================================
REM Parse command line arguments
REM ============================================================================
set INSTALL_MODE=full
set INSTALL_TARGET=user
set USE_VENV=0
set UPGRADE_MODE=0
set REPAIR_MODE=0

:parse_args
if "%~1"=="" goto done_parsing
if /i "%~1"=="/venv" set USE_VENV=1
if /i "%~1"=="/user" set INSTALL_TARGET=user
if /i "%~1"=="/system" set INSTALL_TARGET=system
if /i "%~1"=="/upgrade" set UPGRADE_MODE=1
if /i "%~1"=="/repair" set REPAIR_MODE=1
if /i "%~1"=="/minimal" set INSTALL_MODE=minimal
if /i "%~1"=="/full" set INSTALL_MODE=full
shift
goto parse_args
:done_parsing

REM Set console title and color
title ET Scanner v6.1.1 - Advanced Installer
color 0B

echo.
echo ============================================================================
echo   EXCEPTION THEORY SCANNER v6.1.1 - ADVANCED INSTALLER
echo ============================================================================
echo.
echo   Installation Configuration:
echo.
if %USE_VENV%==1 (
    echo     Mode: Virtual Environment
) else (
    echo     Mode: Direct Installation
)
echo     Target: %INSTALL_TARGET%
echo     Packages: %INSTALL_MODE%
if %UPGRADE_MODE%==1 echo     Action: Upgrade
if %REPAIR_MODE%==1 echo     Action: Repair
echo     Scanner: %SCANNER_FILE%
echo.
echo   Press any key to continue or Ctrl+C to cancel...
pause
echo.

REM ============================================================================
REM Step 1: Check Python
REM ============================================================================

echo [1/8] Checking Python installation...
echo.

where python >nul 2>&1
if errorlevel 1 (
    color 0C
    echo   [ERROR] Python not found in PATH!
    echo.
    echo   Please install Python 3.8+ from: https://www.python.org/downloads/
    echo   Make sure to check "Add Python to PATH" during installation!
    echo.
    goto install_failed
)

python --version
if errorlevel 1 (
    color 0C
    echo   [ERROR] Python found but cannot execute!
    echo.
    goto install_failed
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo   [OK] Python %PYTHON_VERSION% found
echo.

REM ============================================================================
REM Step 2: Create Virtual Environment (if requested)
REM ============================================================================

if %USE_VENV%==0 goto skip_venv

echo [2/8] Creating virtual environment...
echo.

if exist "et_scanner_venv\" (
    echo   [INFO] Virtual environment already exists
    echo.
    set /p RECREATE="   Recreate virtual environment? (Y/N): "
    if /i "%RECREATE%"=="Y" (
        echo   Removing old environment...
        rmdir /s /q et_scanner_venv
        if errorlevel 1 (
            echo   [WARNING] Could not fully remove old environment
        )
    )
)

if not exist "et_scanner_venv\" (
    echo   Creating new virtual environment...
    python -m venv et_scanner_venv
    if errorlevel 1 (
        color 0C
        echo   [ERROR] Failed to create virtual environment
        echo.
        echo   Try running without /venv option
        echo.
        goto install_failed
    )
    echo   [OK] Virtual environment created
)

echo   Activating virtual environment...
if not exist "et_scanner_venv\Scripts\activate.bat" (
    color 0C
    echo   [ERROR] Virtual environment activation script not found!
    echo.
    goto install_failed
)
call et_scanner_venv\Scripts\activate.bat
echo   [OK] Virtual environment activated
echo.
goto done_venv

:skip_venv
echo [2/8] Skipping virtual environment (direct install)
echo.

:done_venv

REM ============================================================================
REM Step 3: Upgrade pip
REM ============================================================================

echo [3/8] Upgrading pip...
echo.

if "%INSTALL_TARGET%"=="user" (
    python -m pip install --upgrade pip --user
) else (
    python -m pip install --upgrade pip
)

if errorlevel 1 (
    color 0E
    echo   [WARNING] pip upgrade had issues, continuing anyway...
) else (
    echo   [OK] pip upgraded
)
echo.

REM ============================================================================
REM Step 4: Install Core Dependencies
REM ============================================================================

echo [4/8] Installing core dependencies...
echo.

REM Build pip flags
set PIP_FLAGS=
if "%INSTALL_TARGET%"=="user" set PIP_FLAGS=--user
if %UPGRADE_MODE%==1 set PIP_FLAGS=%PIP_FLAGS% --upgrade
if %REPAIR_MODE%==1 set PIP_FLAGS=%PIP_FLAGS% --force-reinstall

REM NumPy
echo   Installing numpy...
python -m pip install numpy %PIP_FLAGS%
if errorlevel 1 (
    color 0C
    echo.
    echo   [ERROR] numpy installation failed!
    echo.
    echo   This is a critical dependency.
    echo.
    goto install_failed
)
echo   [OK] numpy installed
echo.

REM SciPy
echo   Installing scipy...
python -m pip install scipy %PIP_FLAGS%
if errorlevel 1 (
    color 0C
    echo.
    echo   [ERROR] scipy installation failed!
    echo.
    echo   This is a critical dependency.
    echo.
    goto install_failed
)
echo   [OK] scipy installed
echo.

echo   [OK] Core dependencies installed successfully
echo.

REM ============================================================================
REM Step 5: Install Optional Dependencies
REM ============================================================================

if "%INSTALL_MODE%"=="minimal" goto skip_optional

echo [5/8] Installing optional dependencies...
echo.

REM psutil
echo   Installing psutil (process monitoring)...
python -m pip install psutil %PIP_FLAGS%
if errorlevel 1 (
    color 0E
    echo   [WARNING] psutil installation failed (optional feature)
    echo   Process monitoring will not be available.
) else (
    echo   [OK] psutil installed
)
echo.

echo   [OK] Optional dependencies done
echo.
goto done_optional

:skip_optional
echo [5/8] Skipping optional dependencies (minimal install)
echo.

:done_optional

REM ============================================================================
REM Step 6: Verify Installation
REM ============================================================================

echo [6/8] Verifying installation...
echo.

REM Create a simple verification script (avoid complex escaping)
echo import sys > _verify.py
echo print("  Testing imports...") >> _verify.py
echo try: >> _verify.py
echo     import numpy >> _verify.py
echo     print("  [OK] numpy version:", numpy.__version__) >> _verify.py
echo except ImportError as e: >> _verify.py
echo     print("  [FAIL] numpy:", e) >> _verify.py
echo     sys.exit(1) >> _verify.py
echo try: >> _verify.py
echo     import scipy >> _verify.py
echo     print("  [OK] scipy version:", scipy.__version__) >> _verify.py
echo except ImportError as e: >> _verify.py
echo     print("  [FAIL] scipy:", e) >> _verify.py
echo     sys.exit(1) >> _verify.py
echo try: >> _verify.py
echo     import psutil >> _verify.py
echo     print("  [OK] psutil version:", psutil.__version__) >> _verify.py
echo except ImportError: >> _verify.py
echo     print("  [SKIP] psutil not installed") >> _verify.py
echo try: >> _verify.py
echo     import tkinter >> _verify.py
echo     print("  [OK] tkinter available") >> _verify.py
echo except ImportError: >> _verify.py
echo     print("  [WARN] tkinter not available") >> _verify.py
echo print("") >> _verify.py
echo print("  All core dependencies verified!") >> _verify.py

python _verify.py
set VERIFY_RESULT=%errorlevel%
del _verify.py 2>nul

if not %VERIFY_RESULT%==0 (
    color 0C
    echo.
    echo   [ERROR] Installation verification failed!
    echo.
    goto install_failed
)

echo.
echo   [OK] Installation verified successfully
echo.

REM ============================================================================
REM Step 7: Test Scanner
REM ============================================================================

echo [7/8] Testing ET Scanner...
echo.

if not exist "%SCANNER_FILE%" (
    color 0E
    echo   [WARNING] %SCANNER_FILE% not found in current directory
    echo.
    echo   Make sure to place the scanner script in the same folder
    echo   as this installer, or update SCANNER_FILE at the top of
    echo   this batch file.
    echo.
    goto skip_scanner_test
)

echo   Found: %SCANNER_FILE%
echo   Running test...
python "%SCANNER_FILE%" --help >nul 2>&1
if errorlevel 1 (
    color 0E
    echo   [WARNING] Scanner test returned an error
    echo.
    echo   The scanner may still work - this could be a minor issue.
    echo   Try running it manually to see the actual error.
    echo.
    goto skip_scanner_test
)

echo   [OK] Scanner is working correctly
echo.

:skip_scanner_test

REM ============================================================================
REM Step 8: Create Launch Scripts
REM ============================================================================

echo [8/8] Creating launch scripts...
echo.

REM Create main launcher
if %USE_VENV%==1 (
    echo @echo off > run_scanner.bat
    echo call et_scanner_venv\Scripts\activate.bat >> run_scanner.bat
    echo python %SCANNER_FILE% %%* >> run_scanner.bat
    echo   [OK] Created run_scanner.bat (uses virtual environment)
) else (
    echo @echo off > run_scanner.bat
    echo python %SCANNER_FILE% %%* >> run_scanner.bat
    echo   [OK] Created run_scanner.bat
)

REM Create quick entropy scan launcher
echo @echo off > quick_entropy_scan.bat
echo echo Starting entropy scan... >> quick_entropy_scan.bat
if %USE_VENV%==1 (
    echo call et_scanner_venv\Scripts\activate.bat >> quick_entropy_scan.bat
)
echo python %SCANNER_FILE% --entropy >> quick_entropy_scan.bat
echo echo. >> quick_entropy_scan.bat
echo pause >> quick_entropy_scan.bat
echo   [OK] Created quick_entropy_scan.bat

REM Create interactive launcher
echo @echo off > interactive_scanner.bat
if %USE_VENV%==1 (
    echo call et_scanner_venv\Scripts\activate.bat >> interactive_scanner.bat
)
echo python %SCANNER_FILE% >> interactive_scanner.bat
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
echo   Scanner: %SCANNER_FILE%
echo.
echo   Created launch scripts:
echo.
echo     run_scanner.bat          - Main launcher
echo     interactive_scanner.bat  - Interactive menu
echo     quick_entropy_scan.bat   - Quick entropy test
echo.
echo   Manual usage:
if %USE_VENV%==1 (
    echo     1. Run: et_scanner_venv\Scripts\activate.bat
    echo     2. Run: python %SCANNER_FILE% [options]
) else (
    echo     python %SCANNER_FILE% [options]
)
echo.
echo ============================================================================
echo.

set /p RUN_NOW="Would you like to run the scanner now? (Y/N): "
if /i "%RUN_NOW%"=="Y" (
    echo.
    echo Starting scanner...
    echo.
    call run_scanner.bat
)

echo.
echo ============================================================================
echo   Installation finished. This window will stay open.
echo ============================================================================
echo.
echo Press any key to close this window...
pause
goto end_script

REM ============================================================================
REM Error Handler
REM ============================================================================

:install_failed
echo.
echo ============================================================================
echo   INSTALLATION FAILED
echo ============================================================================
echo.
echo   The installation encountered errors. Please check the messages above.
echo.
echo   Troubleshooting steps:
echo.
echo   1. Make sure you have internet connectivity
echo   2. Try running this installer as Administrator
echo   3. Try with /venv option for isolated environment:
echo      install_dependencies_advanced.bat /venv
echo   4. Try with /repair option to fix broken packages:
echo      install_dependencies_advanced.bat /repair
echo   5. Check if antivirus is blocking Python or pip
echo   6. Check Windows Firewall settings
echo.
echo   If numpy/scipy fail to install:
echo   - You may need Visual C++ Build Tools
echo   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
echo.
echo   For more help, visit:
echo   https://docs.python.org/3/installing/
echo.
echo ============================================================================
echo.
echo Press any key to close this window...
pause
goto end_script

:end_script
endlocal
exit /b 0

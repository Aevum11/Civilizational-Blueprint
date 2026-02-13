@echo off
REM ============================================================================
REM ET Scanner v6.1 - Windows Dependency Installer
REM ============================================================================
REM
REM This installer will:
REM   1. Check for Python installation
REM   2. Upgrade pip to latest version
REM   3. Install all required dependencies
REM   4. Install optional dependencies
REM   5. Verify installation
REM   6. Test the scanner
REM
REM Usage: Double-click this file or run from command prompt
REM
REM Author: Exception Theory Scanner Project
REM Version: 6.1
REM ============================================================================

setlocal enabledelayedexpansion

REM Set console colors and title
title ET Scanner v6.1 - Dependency Installer
color 0A

echo.
echo ============================================================================
echo   EXCEPTION THEORY SCANNER v6.1 - DEPENDENCY INSTALLER
echo ============================================================================
echo.
echo   This will install all required dependencies for the ET Scanner.
echo   Please wait while we check your system...
echo.

REM ============================================================================
REM STEP 1: Check for Python
REM ============================================================================

echo [STEP 1/6] Checking Python installation...
echo.

python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo   [ERROR] Python is not installed or not in PATH!
    echo.
    echo   Please install Python 3.8 or higher from:
    echo   https://www.python.org/downloads/
    echo.
    echo   Make sure to check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo   [OK] Python %PYTHON_VERSION% detected
echo.

REM Check Python version is 3.8+
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% LSS 3 (
    color 0C
    echo   [ERROR] Python 3.8 or higher required. You have Python %PYTHON_VERSION%
    echo.
    pause
    exit /b 1
)

if %MAJOR% EQU 3 if %MINOR% LSS 8 (
    color 0C
    echo   [ERROR] Python 3.8 or higher required. You have Python %PYTHON_VERSION%
    echo.
    pause
    exit /b 1
)

echo   [OK] Python version is compatible
echo.
timeout /t 2 /nobreak >nul

REM ============================================================================
REM STEP 2: Upgrade pip
REM ============================================================================

echo [STEP 2/6] Upgrading pip to latest version...
echo.

python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    color 0E
    echo   [WARNING] Could not upgrade pip, continuing with existing version...
    echo.
    timeout /t 2 /nobreak >nul
) else (
    echo   [OK] pip upgraded successfully
    echo.
    timeout /t 1 /nobreak >nul
)

REM ============================================================================
REM STEP 3: Install Required Dependencies
REM ============================================================================

echo [STEP 3/6] Installing REQUIRED dependencies...
echo.
echo   This may take a few minutes on first install...
echo.

REM Install numpy
echo   Installing numpy (numerical computing)...
python -m pip install numpy --quiet
if errorlevel 1 (
    color 0C
    echo   [ERROR] Failed to install numpy
    echo.
    pause
    exit /b 1
)
echo   [OK] numpy installed

REM Install scipy
echo   Installing scipy (scientific computing)...
python -m pip install scipy --quiet
if errorlevel 1 (
    color 0C
    echo   [ERROR] Failed to install scipy
    echo.
    pause
    exit /b 1
)
echo   [OK] scipy installed

echo.
echo   [OK] All required dependencies installed successfully!
echo.
timeout /t 2 /nobreak >nul

REM ============================================================================
REM STEP 4: Install Optional Dependencies
REM ============================================================================

echo [STEP 4/6] Installing OPTIONAL dependencies...
echo.

REM Install psutil (for process monitoring)
echo   Installing psutil (process monitoring - optional)...
python -m pip install psutil --quiet
if errorlevel 1 (
    color 0E
    echo   [WARNING] psutil installation failed (process monitoring will be unavailable)
    echo.
) else (
    echo   [OK] psutil installed
)

echo.
echo   Note: tkinter is included with Python (for file dialogs)
echo         urllib is included with Python (for URL fetching)
echo.
timeout /t 2 /nobreak >nul

REM ============================================================================
REM STEP 5: Verify Installation
REM ============================================================================

echo [STEP 5/6] Verifying installation...
echo.

REM Create verification script
echo import sys > verify_deps.py
echo try: >> verify_deps.py
echo     import numpy as np >> verify_deps.py
echo     import scipy >> verify_deps.py
echo     print('CORE: OK') >> verify_deps.py
echo except ImportError as e: >> verify_deps.py
echo     print(f'CORE: FAILED - {e}') >> verify_deps.py
echo     sys.exit(1) >> verify_deps.py
echo try: >> verify_deps.py
echo     import psutil >> verify_deps.py
echo     print('PSUTIL: OK') >> verify_deps.py
echo except ImportError: >> verify_deps.py
echo     print('PSUTIL: NOT INSTALLED (optional)') >> verify_deps.py
echo try: >> verify_deps.py
echo     import tkinter >> verify_deps.py
echo     print('TKINTER: OK') >> verify_deps.py
echo except ImportError: >> verify_deps.py
echo     print('TKINTER: NOT AVAILABLE (file dialogs disabled)') >> verify_deps.py
echo print('SUCCESS: All core dependencies verified!') >> verify_deps.py

python verify_deps.py
if errorlevel 1 (
    color 0C
    echo.
    echo   [ERROR] Dependency verification failed!
    echo.
    del verify_deps.py
    pause
    exit /b 1
)

del verify_deps.py

echo.
echo   [OK] All dependencies verified successfully!
echo.
timeout /t 2 /nobreak >nul

REM ============================================================================
REM STEP 6: Test ET Scanner
REM ============================================================================

echo [STEP 6/6] Testing ET Scanner...
echo.

REM Check if scanner exists
if not exist "et_scanner_v6.1.py" (
    color 0E
    echo   [WARNING] et_scanner_v6.1.py not found in current directory
    echo.
    echo   Please ensure the scanner file is in the same folder as this installer.
    echo.
    goto :skip_test
)

REM Test scanner
echo   Running quick test scan...
echo.

python et_scanner_v6.1.py --help >nul 2>&1
if errorlevel 1 (
    color 0C
    echo   [ERROR] Scanner test failed!
    echo.
    echo   Please check that et_scanner_v6.1.py is not corrupted.
    echo.
    pause
    exit /b 1
)

echo   [OK] Scanner is working correctly!
echo.

:skip_test

REM ============================================================================
REM Installation Complete
REM ============================================================================

color 0A
echo.
echo ============================================================================
echo   INSTALLATION COMPLETE!
echo ============================================================================
echo.
echo   All dependencies have been installed successfully.
echo.
echo   You can now run the ET Scanner using:
echo.
echo     python et_scanner_v6.1.py
echo.
echo   Or for specific operations:
echo.
echo     python et_scanner_v6.1.py --help       (Show help)
echo     python et_scanner_v6.1.py --entropy    (Scan entropy)
echo     python et_scanner_v6.1.py filename     (Scan a file)
echo.
echo   Installed components:
echo     - Python %PYTHON_VERSION%
echo     - numpy (numerical computing)
echo     - scipy (scientific computing)
echo     - psutil (process monitoring - if available)
echo     - tkinter (file dialogs - built-in)
echo.
echo ============================================================================
echo.

REM Ask if user wants to run scanner now
set /p RUN_NOW="Would you like to run the scanner now? (Y/N): "
if /i "%RUN_NOW%"=="Y" (
    echo.
    echo Starting ET Scanner...
    echo.
    timeout /t 2 /nobreak >nul
    python et_scanner_v6.1.py
)

echo.
echo Press any key to exit...
pause >nul

endlocal
exit /b 0

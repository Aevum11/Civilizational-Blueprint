@echo off
REM ============================================================================
REM ET Scanner v6.1 - Quick Launcher
REM ============================================================================
REM
REM Simple launcher for daily use.
REM Just double-click to run the ET Scanner!
REM
REM ============================================================================

title ET Scanner v6.1

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo.
    echo ERROR: Python not found!
    echo.
    echo Please install Python or run install_dependencies.bat first.
    echo.
    pause
    exit /b 1
)

REM Check if scanner exists
if not exist "et_scanner_v6.1.py" (
    color 0C
    echo.
    echo ERROR: et_scanner_v6.1.py not found!
    echo.
    echo Please ensure the scanner file is in the same directory as this launcher.
    echo.
    pause
    exit /b 1
)

REM Launch scanner
cls
python et_scanner_v6.1.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo.
    echo Scanner exited with an error.
    echo.
    pause
)

exit /b 0

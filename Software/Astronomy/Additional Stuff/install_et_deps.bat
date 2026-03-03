@echo off
title ET Scanner Environment Installer
color 0A

echo ========================================================
echo [HARMONY] Initializing Substrate Configuration...
echo ========================================================
echo.

:: 1. Check if Python is reachable
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [CRITICAL] Python not found in PATH.
    echo Please install Python 3.8+ and check "Add to PATH" in the installer.
    pause
    exit /b
)

echo [IDENTITY] Python detected. Updating pip...
echo.
python -m pip install --upgrade pip

echo.
echo [MANIFOLD] Installing required neural pathways...
echo --------------------------------------------------------

:: 2. Install Dependencies using -m pip
:: requests: For NetworkGateway and Webhooks
:: watchdog: For Live Surveillance (The Watcher)
:: tqdm: For Progress Bars (Visual Harmony)
python -m pip install requests watchdog tqdm

if %errorlevel% neq 0 (
    echo.
    echo [CHAOS] Installation encountered an error.
    echo Check your internet connection or try running as Administrator.
    pause
    exit /b
)

echo.
echo ========================================================
echo [PEACE] Environment Substantiated.
echo You may now run: python et_harmony.py
echo ========================================================
pause
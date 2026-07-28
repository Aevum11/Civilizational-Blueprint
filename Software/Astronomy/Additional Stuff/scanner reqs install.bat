@echo off
TITLE ET Manifold Scanner - Dependency Installer

echo ========================================================
echo   ET MANIFOLD SCANNER DEPENDENCY INSTALLER
echo   Method: Direct Python Invocation (python -m pip)
echo ========================================================
echo.

:: 1. Check if Python is accessible at all
echo [STATUS] Checking for Python executable...
python --version
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 'python' command not found. Please ensure Python is installed and added to PATH.
    pause
    exit /b
)

:: 2. Upgrade PIP itself first (fixes many installation header issues)
echo.
echo [STATUS] Upgrading pip internal core...
python -m pip install --upgrade pip

:: 3. Install the Science Stack
echo.
echo [STATUS] Installing Numerical and Image Processing libraries...
:: numpy: Matrix math
:: scipy: ndimage filters (Gaussian, Median, Uniform)
:: astropy: FITS file handling (Required for the Scanner I/O)
python -m pip install numpy scipy astropy

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Installation failed. Please check the error messages above.
) ELSE (
    echo.
    echo [SUCCESS] All libraries installed successfully.
    echo          - numpy
    echo          - scipy
    echo          - astropy
)

echo.
pause
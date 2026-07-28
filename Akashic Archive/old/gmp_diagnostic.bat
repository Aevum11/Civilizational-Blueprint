@echo off
setlocal enabledelayedexpansion

set LOGFILE=%~dp0gmp_diagnostic.txt
echo GMP Build Diagnostic — %date% %time% > "%LOGFILE%"

echo ===================================================
echo  GMP Build Diagnostic
echo  Dumping ALL log files from C:\vcpkg\buildtrees\gmp
echo  Output: %LOGFILE%
echo ===================================================
echo.

:: ── List every file in the buildtrees/gmp directory ──
echo === ALL FILES IN buildtrees\gmp === >> "%LOGFILE%"
echo === ALL FILES IN buildtrees\gmp ===
if exist "C:\vcpkg\buildtrees\gmp" (
    dir /s /b "C:\vcpkg\buildtrees\gmp\*.log" >> "%LOGFILE%" 2>&1
    dir /s /b "C:\vcpkg\buildtrees\gmp\*.log"
    echo.
    echo. >> "%LOGFILE%"

    echo === DUMPING EVERY .log FILE === >> "%LOGFILE%"
    echo === DUMPING EVERY .log FILE ===
    echo.

    for /r "C:\vcpkg\buildtrees\gmp" %%f in (*.log) do (
        echo. >> "%LOGFILE%"
        echo ──────────────────────────────────────── >> "%LOGFILE%"
        echo FILE: %%f >> "%LOGFILE%"
        echo ──────────────────────────────────────── >> "%LOGFILE%"
        echo   Dumping: %%f
        type "%%f" >> "%LOGFILE%"
    )
) else (
    echo   C:\vcpkg\buildtrees\gmp does NOT exist
    echo   C:\vcpkg\buildtrees\gmp does NOT exist >> "%LOGFILE%"
    echo   The build artifacts may have been cleaned.
    echo   The build artifacts may have been cleaned. >> "%LOGFILE%"
    echo.
    echo   Trying a fresh GMP build to capture logs...
    echo   Trying a fresh GMP build... >> "%LOGFILE%"
    
    cd /d C:\vcpkg
    set VCPKG_MAX_CONCURRENCY=1
    .\vcpkg install gmp:x64-windows-static --no-binarycaching >> "%LOGFILE%" 2>&1
    
    echo. >> "%LOGFILE%"
    echo === POST-BUILD: ALL FILES === >> "%LOGFILE%"
    if exist "C:\vcpkg\buildtrees\gmp" (
        dir /s /b "C:\vcpkg\buildtrees\gmp\*.log" >> "%LOGFILE%" 2>&1
        for /r "C:\vcpkg\buildtrees\gmp" %%f in (*.log) do (
            echo. >> "%LOGFILE%"
            echo ──────────────────────────────────────── >> "%LOGFILE%"
            echo FILE: %%f >> "%LOGFILE%"
            echo ──────────────────────────────────────── >> "%LOGFILE%"
            type "%%f" >> "%LOGFILE%"
        )
    )
)

:: ── Also check vcpkg's own install log ───────────────
echo. >> "%LOGFILE%"
echo === vcpkg install-log directory === >> "%LOGFILE%"
if exist "C:\vcpkg\installed\vcpkg\updates" (
    dir /s /b "C:\vcpkg\installed\vcpkg\updates\*" >> "%LOGFILE%" 2>&1
)

echo.
echo ===================================================
echo  Done. Upload %LOGFILE% to Claude.
echo ===================================================
echo.
pause

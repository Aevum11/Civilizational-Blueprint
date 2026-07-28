@echo off
setlocal enabledelayedexpansion

echo Renaming all .py files to .txt in this folder and all subfolders...
echo.

for /r %%F in (*.py) do (
    set "old=%%F"
    set "new=%%~dpF%%~nF.txt"
    
    if exist "!new!" (
        echo WARNING: Skipping "%%~nxF" - a file named "%%~nF.txt" already exists
    ) else (
        ren "%%F" "%%~nF.txt"
        echo Renamed: "%%~nxF" --^> "%%~nF.txt"
    )
)

echo.
echo Done! Press any key to exit...
pause >nul
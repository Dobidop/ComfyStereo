@echo off
echo Cloning ComfyStereo repository and replacing current directory contents...

REM Check if git is available
git --version >nul 2>&1
if errorlevel 1 (
    echo Error: Git is not installed or not in PATH
    pause
    exit /b 1
)

REM Clone to temporary directory with specific branch
echo Cloning repository branch: claude/review-codebase-bugs-01Dh7NHVv4uypSYkgYCdmqjy...
git clone -b claude/review-codebase-bugs-01Dh7NHVv4uypSYkgYCdmqjy https://github.com/Dobidop/ComfyStereo.git temp_clone
if errorlevel 1 (
    echo Error: Failed to clone repository
    pause
    exit /b 1
)

REM Remove existing files (but keep this batch file)
echo Removing existing files...
for /f "delims=" %%i in ('dir /b /a-d ^| findstr /v /i "cloneComfyStereo.bat"') do (
    del "%%i" 2>nul
)

REM Remove existing directories
for /f "delims=" %%i in ('dir /b /ad ^| findstr /v /i "temp_clone"') do (
    rmdir /s /q "%%i" 2>nul
)

REM Move files from temp_clone to current directory
echo Moving files...
xcopy "temp_clone\*" "." /e /h /y
if errorlevel 1 (
    echo Warning: Some files may not have been copied
)

REM Clean up temporary directory
echo Cleaning up...
rmdir /s /q temp_clone

echo Done! ComfyStereo has been cloned and contents moved to current directory.
pause
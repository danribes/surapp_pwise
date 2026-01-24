@echo off
REM SURAPP Docker Runner for Windows
REM
REM This script simplifies running SURAPP in Docker.
REM
REM Usage:
REM   run.bat <image_file> [options]
REM
REM Examples:
REM   run.bat my_plot.png
REM   run.bat my_plot.png --time-max 24
REM   run.bat my_plot.png --curves 3 --time-max 36

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

REM Check if Docker is installed
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Docker is not installed.
    echo Please install Docker from https://www.docker.com/get-started
    exit /b 1
)

REM Check if image argument provided
if "%~1"=="" (
    echo SURAPP - Kaplan-Meier Curve Extractor
    echo.
    echo Usage: %~nx0 ^<image_file^> [options]
    echo.
    echo Options:
    echo   --time-max TIME    Maximum time value on X-axis
    echo   --curves N         Expected number of curves ^(default: 2^)
    echo   -o, --output DIR   Output directory
    echo.
    echo Examples:
    echo   %~nx0 my_km_plot.png
    echo   %~nx0 my_km_plot.png --time-max 24
    echo   %~nx0 my_km_plot.png --curves 2 --time-max 36
    exit /b 0
)

set "IMAGE_FILE=%~1"
shift

REM Check if image file exists
if not exist "%IMAGE_FILE%" (
    echo Error: Image file not found: %IMAGE_FILE%
    exit /b 1
)

REM Get absolute path and filename
for %%F in ("%IMAGE_FILE%") do (
    set "IMAGE_DIR=%%~dpF"
    set "IMAGE_NAME=%%~nxF"
)

REM Remove trailing backslash from IMAGE_DIR
if "%IMAGE_DIR:~-1%"=="\" set "IMAGE_DIR=%IMAGE_DIR:~0,-1%"

REM Create output directory
set "OUTPUT_DIR=%IMAGE_DIR%\results"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo SURAPP - Kaplan-Meier Curve Extractor
echo ========================================
echo Input:  %IMAGE_FILE%
echo Output: %OUTPUT_DIR%
echo.

REM Build Docker image if not exists
docker image inspect surapp:latest >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Building Docker image ^(first run only^)...
    docker build -t surapp:latest -f "%SCRIPT_DIR%Dockerfile" "%PROJECT_ROOT%"
    echo.
)

REM Collect remaining arguments
set "EXTRA_ARGS="
:argloop
if "%~1"=="" goto argdone
set "EXTRA_ARGS=!EXTRA_ARGS! %~1"
shift
goto argloop
:argdone

REM Run the extraction
echo Running extraction...
echo.

docker run --rm ^
    -v "%IMAGE_DIR%:/data/input:ro" ^
    -v "%OUTPUT_DIR%:/data/output" ^
    surapp:latest ^
    python /app/extract_km.py "/data/input/%IMAGE_NAME%" -o "/data/output" %EXTRA_ARGS%

echo.
echo Done! Results saved to: %OUTPUT_DIR%

endlocal

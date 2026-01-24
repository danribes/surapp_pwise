@echo off
REM SURAPP AI-Enhanced Docker Runner for Windows
REM
REM This script runs SURAPP with AI-powered validation using Ollama.
REM
REM Usage:
REM   run-ai.bat <image_file> [options]
REM
REM Examples:
REM   run-ai.bat my_plot.png --validate
REM   run-ai.bat --status

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

REM Handle special commands
if "%~1"=="--status" goto :status
if "%~1"=="--start" goto :start
if "%~1"=="--stop" goto :stop
if "%~1"=="--help" goto :help
if "%~1"=="-h" goto :help
if "%~1"=="" goto :help

REM Process image file
set "IMAGE_FILE=%~1"
shift

REM Check if image file exists
if not exist "%IMAGE_FILE%" (
    echo Error: Image file not found: %IMAGE_FILE%
    exit /b 1
)

REM Check if Ollama is running
docker ps --format "{{.Names}}" | findstr /C:"surapp-ollama" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo AI services not running. Starting...
    call :start_services
)

REM Get absolute path and filename
for %%F in ("%IMAGE_FILE%") do (
    set "IMAGE_DIR=%%~dpF"
    set "IMAGE_NAME=%%~nxF"
)

REM Remove trailing backslash
if "%IMAGE_DIR:~-1%"=="\" set "IMAGE_DIR=%IMAGE_DIR:~0,-1%"

REM Create output directory
set "OUTPUT_DIR=%IMAGE_DIR%\results"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo SURAPP AI-Enhanced - Kaplan-Meier Curve Extractor
echo ==================================================
echo Input:  %IMAGE_FILE%
echo Output: %OUTPUT_DIR%
echo.

REM Build AI image if not exists
docker image inspect surapp-ai:latest >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Building AI-enhanced Docker image ^(first run only^)...
    cd /d "%SCRIPT_DIR%"
    docker-compose -f docker-compose.yml -f docker-compose.ai.yml build surapp-ai
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

REM Run AI-enhanced extraction
echo Running AI-enhanced extraction...
echo.

docker run --rm ^
    --network host ^
    -e OLLAMA_HOST=http://localhost:11434 ^
    -e AI_MODEL=llama3.2-vision ^
    -e AI_ENABLED=true ^
    -v "%IMAGE_DIR%:/data/input:ro" ^
    -v "%OUTPUT_DIR%:/data/output" ^
    surapp-ai:latest ^
    python /app/extract_km_ai.py "/data/input/%IMAGE_NAME%" -o "/data/output" %EXTRA_ARGS%

echo.
echo Done! Results saved to: %OUTPUT_DIR%
goto :eof

:status
echo SURAPP AI Service Status
echo =========================
docker ps --format "{{.Names}}" | findstr /C:"surapp-ollama" >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo Ollama Container: Running
    curl -s http://localhost:11434/api/tags >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        echo Ollama API: Available
    ) else (
        echo Ollama API: Not responding
    )
) else (
    echo Ollama Container: Not running
    echo.
    echo Start with:
    echo   run-ai.bat --start
)
goto :eof

:start
:start_services
echo Starting AI services...
cd /d "%SCRIPT_DIR%"
docker-compose -f docker-compose.yml -f docker-compose.ai.yml up -d ollama

echo Waiting for Ollama to be ready...
set /a count=0
:wait_loop
if %count% geq 30 (
    echo Timeout waiting for Ollama
    goto :eof
)
curl -s http://localhost:11434/api/tags >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo Ollama is ready!
    goto :ensure_model
)
timeout /t 2 /nobreak >nul
set /a count+=1
goto :wait_loop

:ensure_model
echo Checking for llama3.2-vision model...
docker exec surapp-ollama ollama list | findstr /C:"llama3.2-vision" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Pulling llama3.2-vision ^(this may take a while, ~4GB^)...
    docker exec surapp-ollama ollama pull llama3.2-vision
)
echo Model ready!
goto :eof

:stop
echo Stopping AI services...
cd /d "%SCRIPT_DIR%"
docker-compose -f docker-compose.yml -f docker-compose.ai.yml down
echo AI services stopped
goto :eof

:help
echo SURAPP AI-Enhanced - Kaplan-Meier Curve Extractor
echo.
echo Usage: %~nx0 ^<image_file^> [options]
echo        %~nx0 --status    # Check AI service status
echo        %~nx0 --start     # Start AI services
echo        %~nx0 --stop      # Stop AI services
echo.
echo Options:
echo   --time-max TIME    Maximum time value on X-axis
echo   --curves N         Expected number of curves ^(default: 2^)
echo   --validate         Enable AI validation
echo   -o, --output DIR   Output directory
echo.
echo Examples:
echo   %~nx0 --start                              # Start AI services first
echo   %~nx0 my_km_plot.png --validate            # Extract with AI validation
echo   %~nx0 my_km_plot.png --time-max 24 --validate
goto :eof

endlocal

@echo off
REM SURAPP - Unified Entry Point for Windows
REM
REM This script provides a single entry point for all SURAPP execution modes:
REM   1. Standard (Python)  - Direct Python execution
REM   2. Standard (Docker)  - Containerized execution
REM   3. AI-Enhanced        - With llama3.2-vision validation
REM
REM Usage:
REM   surapp.bat [image_file] [options]
REM   surapp.bat --mode python image.png
REM   surapp.bat --mode docker image.png
REM   surapp.bat --mode ai image.png

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"
set "MODE="
set "IMAGE_FILE="
set "EXTRA_ARGS="

REM Parse arguments
:parse_args
if "%~1"=="" goto :done_parsing

if "%~1"=="--mode" (
    set "MODE=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help
if "%~1"=="--status" goto :show_status
if "%~1"=="--start" goto :start_ai
if "%~1"=="--stop" goto :stop_ai

REM Check if argument starts with -
echo %~1 | findstr /B "-" >nul
if %ERRORLEVEL% equ 0 (
    set "EXTRA_ARGS=!EXTRA_ARGS! %~1"
    shift
    goto :parse_args
)

REM First non-flag is image file
if "!IMAGE_FILE!"=="" (
    set "IMAGE_FILE=%~1"
) else (
    set "EXTRA_ARGS=!EXTRA_ARGS! %~1"
)
shift
goto :parse_args

:done_parsing

REM Show banner
call :show_banner

REM Check if image file provided
if "!IMAGE_FILE!"=="" (
    echo No image file specified.
    echo.
    echo Usage: %~nx0 [--mode python^|docker^|ai] ^<image_file^> [options]
    echo.
    echo Run '%~nx0 --help' for more information.
    echo Run '%~nx0 --status' to check system status.
    exit /b 1
)

REM Check if image file exists
if not exist "!IMAGE_FILE!" (
    echo Error: Image file not found: !IMAGE_FILE!
    exit /b 1
)

REM If mode not specified, ask interactively
if "!MODE!"=="" (
    call :select_mode_interactive
    if "!MODE!"=="" exit /b 0
)

REM Validate and run mode
if /i "!MODE!"=="python" goto :run_python
if /i "!MODE!"=="py" goto :run_python
if /i "!MODE!"=="native" goto :run_python
if /i "!MODE!"=="docker" goto :run_docker
if /i "!MODE!"=="container" goto :run_docker
if /i "!MODE!"=="ai" goto :run_ai
if /i "!MODE!"=="ai-docker" goto :run_ai
if /i "!MODE!"=="validate" goto :run_ai

echo Error: Unknown mode '!MODE!'
echo Valid modes: python, docker, ai
exit /b 1

REM ============================================
REM Functions
REM ============================================

:show_banner
echo.
echo ===============================================================
echo            SURAPP - Kaplan-Meier Curve Extractor
echo ===============================================================
echo.
goto :eof

:show_help
call :show_banner
echo Usage: %~nx0 [options] [image_file] [extraction_options]
echo.
echo Options:
echo   --mode MODE    Execution mode: python, docker, or ai
echo   --help, -h     Show this help message
echo   --status       Check system status (Docker, AI services)
echo.
echo Execution Modes:
echo   python   Run directly with Python (fastest, requires Python setup)
echo   docker   Run in Docker container (no Python setup needed)
echo   ai       Run with AI validation (requires Docker + ~4GB model)
echo.
echo Extraction Options (passed to extractor):
echo   --time-max TIME    Maximum time value on X-axis
echo   --curves N         Expected number of curves (default: 2)
echo   -o, --output DIR   Output directory
echo   --validate         Enable AI validation (ai mode only)
echo.
echo Examples:
echo   %~nx0 my_plot.png                    # Interactive mode selection
echo   %~nx0 --mode python my_plot.png      # Use Python directly
echo   %~nx0 --mode docker my_plot.png      # Use Docker
echo   %~nx0 --mode ai my_plot.png          # Use AI validation
echo   %~nx0 my_plot.png --time-max 36      # With extraction options
echo.
exit /b 0

:show_status
call :show_banner
echo System Status
echo =============
echo.

REM Check Python
echo Python:
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo   Available: Yes
    python -c "import cv2, numpy, pandas, matplotlib" 2>nul
    if %ERRORLEVEL% equ 0 (
        echo   Dependencies: Installed
    ) else (
        echo   Dependencies: Missing ^(run: pip install -r requirements.txt^)
    )
) else (
    echo   Available: No
)

REM Check Docker
echo.
echo Docker:
where docker >nul 2>nul
if %ERRORLEVEL% equ 0 (
    docker info >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        echo   Available: Yes
    ) else (
        echo   Available: No ^(Docker not running^)
    )
) else (
    echo   Available: No
)

REM Check AI
echo.
echo AI Services:
curl -s http://localhost:11434/api/tags >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo   Ollama: Running
) else (
    echo   Ollama: Not running
)

echo.
echo Available Modes:
echo   [1] python  - Run with native Python
echo   [2] docker  - Run in Docker container
echo   [3] ai      - Run with AI validation
echo.
exit /b 0

:select_mode_interactive
echo.
echo Select execution mode:
echo.
echo   [1] Python (Native)     - Fastest startup
echo   [2] Docker (Standard)   - No Python setup needed
echo   [3] Docker (AI)         - With AI validation
echo.
echo   [0] Cancel
echo.

:select_loop
set /p choice="Select mode [1-3]: "
if "%choice%"=="1" (
    set "MODE=python"
    goto :eof
)
if "%choice%"=="2" (
    set "MODE=docker"
    goto :eof
)
if "%choice%"=="3" (
    set "MODE=ai"
    goto :eof
)
if "%choice%"=="0" (
    echo Cancelled.
    set "MODE="
    goto :eof
)
echo Please enter 1, 2, 3, or 0 to cancel
goto :select_loop

:run_python
echo Mode: python
echo Image: !IMAGE_FILE!
if not "!EXTRA_ARGS!"=="" echo Options: !EXTRA_ARGS!
echo.

REM Check Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python not found
    exit /b 1
)

echo Running with Python...
echo.
cd /d "%PROJECT_ROOT%"
python src\extract_km.py "!IMAGE_FILE!" !EXTRA_ARGS!
exit /b %ERRORLEVEL%

:run_docker
echo Mode: docker
echo Image: !IMAGE_FILE!
if not "!EXTRA_ARGS!"=="" echo Options: !EXTRA_ARGS!
echo.

REM Check Docker
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Docker not found
    exit /b 1
)

echo Running with Docker...
echo.
cd /d "%PROJECT_ROOT%"
call docker\run.bat "!IMAGE_FILE!" !EXTRA_ARGS!
exit /b %ERRORLEVEL%

:run_ai
echo Mode: ai
echo Image: !IMAGE_FILE!
if not "!EXTRA_ARGS!"=="" echo Options: !EXTRA_ARGS!
echo.

REM Check Docker
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Docker not found ^(required for AI mode^)
    exit /b 1
)

REM Add --validate if not present
echo !EXTRA_ARGS! | findstr /C:"--validate" >nul
if %ERRORLEVEL% neq 0 (
    set "EXTRA_ARGS=!EXTRA_ARGS! --validate"
)

echo Running with AI validation...
echo.
cd /d "%PROJECT_ROOT%"
call docker\run-ai.bat "!IMAGE_FILE!" !EXTRA_ARGS!
exit /b %ERRORLEVEL%

:start_ai
cd /d "%PROJECT_ROOT%"
call docker\run-ai.bat --start
exit /b 0

:stop_ai
cd /d "%PROJECT_ROOT%"
call docker\run-ai.bat --stop
exit /b 0

endlocal

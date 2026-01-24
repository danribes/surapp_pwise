#!/bin/bash
# SURAPP - Unified Entry Point
#
# This script provides a single entry point for all SURAPP execution modes:
#   1. Standard (Python)  - Direct Python execution
#   2. Standard (Docker)  - Containerized execution
#   3. AI-Enhanced        - With llama3.2-vision validation
#
# Usage:
#   ./surapp.sh [image_file] [options]
#   ./surapp.sh --mode python image.png
#   ./surapp.sh --mode docker image.png
#   ./surapp.sh --mode ai image.png

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
MODE=""
IMAGE_FILE=""
EXTRA_ARGS=""

# Function to show banner
show_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║           SURAPP - Kaplan-Meier Curve Extractor           ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Function to show help
show_help() {
    show_banner
    echo "Usage: $0 [options] [image_file] [extraction_options]"
    echo ""
    echo "Options:"
    echo "  --mode MODE    Execution mode: python, docker, or ai"
    echo "  --help, -h     Show this help message"
    echo "  --status       Check system status (Docker, AI services)"
    echo ""
    echo "Execution Modes:"
    echo "  python   Run directly with Python (fastest, requires Python setup)"
    echo "  docker   Run in Docker container (no Python setup needed)"
    echo "  ai       Run with AI validation (requires Docker + ~4GB model)"
    echo ""
    echo "Extraction Options (passed to extractor):"
    echo "  --time-max TIME    Maximum time value on X-axis"
    echo "  --curves N         Expected number of curves (default: 2)"
    echo "  -o, --output DIR   Output directory"
    echo "  --validate         Enable AI validation (ai mode only)"
    echo ""
    echo "Examples:"
    echo "  $0 my_plot.png                    # Interactive mode selection"
    echo "  $0 --mode python my_plot.png      # Use Python directly"
    echo "  $0 --mode docker my_plot.png      # Use Docker"
    echo "  $0 --mode ai my_plot.png          # Use AI validation"
    echo "  $0 my_plot.png --time-max 36      # With extraction options"
    echo ""
}

# Function to check Python availability
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        return 0
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        return 0
    fi
    return 1
}

# Function to check Python dependencies
check_python_deps() {
    $PYTHON_CMD -c "import cv2, numpy, pandas, matplotlib" 2>/dev/null
    return $?
}

# Function to check Docker availability
check_docker() {
    command -v docker &> /dev/null && docker info &> /dev/null
    return $?
}

# Function to check if Ollama is running
check_ollama() {
    curl -s http://localhost:11434/api/tags &> /dev/null
    return $?
}

# Function to show system status
show_status() {
    show_banner
    echo -e "${BOLD}System Status${NC}"
    echo "============="
    echo ""

    # Python
    echo -n "Python:           "
    if check_python; then
        echo -e "${GREEN}Available${NC} ($($PYTHON_CMD --version 2>&1))"
        echo -n "  Dependencies:   "
        if check_python_deps; then
            echo -e "${GREEN}Installed${NC}"
        else
            echo -e "${YELLOW}Missing${NC} (run: pip install -r requirements.txt)"
        fi
    else
        echo -e "${RED}Not found${NC}"
    fi

    # Docker
    echo -n "Docker:           "
    if check_docker; then
        echo -e "${GREEN}Available${NC}"

        # Check for SURAPP images
        echo -n "  surapp image:   "
        if docker image inspect surapp:latest &> /dev/null; then
            echo -e "${GREEN}Built${NC}"
        else
            echo -e "${YELLOW}Not built${NC} (will build on first run)"
        fi

        echo -n "  surapp-ai:      "
        if docker image inspect surapp-ai:latest &> /dev/null; then
            echo -e "${GREEN}Built${NC}"
        else
            echo -e "${YELLOW}Not built${NC} (will build on first run)"
        fi
    else
        echo -e "${RED}Not available${NC}"
    fi

    # Ollama/AI
    echo -n "AI Services:      "
    if check_ollama; then
        echo -e "${GREEN}Running${NC}"
        echo -n "  llama3.2-vision: "
        if curl -s http://localhost:11434/api/tags | grep -q "llama3.2-vision"; then
            echo -e "${GREEN}Installed${NC}"
        else
            echo -e "${YELLOW}Not installed${NC} (run: ./surapp.sh --mode ai --start)"
        fi
    else
        echo -e "${YELLOW}Not running${NC}"
    fi

    echo ""
    echo -e "${BOLD}Available Modes${NC}"
    echo "==============="

    # Python mode
    echo -n "  [1] python:  "
    if check_python && check_python_deps; then
        echo -e "${GREEN}Ready${NC}"
    else
        echo -e "${RED}Not ready${NC}"
    fi

    # Docker mode
    echo -n "  [2] docker:  "
    if check_docker; then
        echo -e "${GREEN}Ready${NC}"
    else
        echo -e "${RED}Not ready${NC}"
    fi

    # AI mode
    echo -n "  [3] ai:      "
    if check_docker; then
        if check_ollama; then
            echo -e "${GREEN}Ready${NC}"
        else
            echo -e "${YELLOW}Ready${NC} (AI services will start automatically)"
        fi
    else
        echo -e "${RED}Not ready${NC} (requires Docker)"
    fi

    echo ""
}

# Function to select mode interactively
select_mode_interactive() {
    echo ""
    echo -e "${BOLD}Select execution mode:${NC}"
    echo ""

    # Check availability and show options
    local python_status docker_status ai_status

    if check_python && check_python_deps; then
        python_status="${GREEN}Ready${NC}"
    else
        python_status="${RED}Not available${NC}"
    fi

    if check_docker; then
        docker_status="${GREEN}Ready${NC}"
        ai_status="${GREEN}Ready${NC}"
    else
        docker_status="${RED}Not available${NC}"
        ai_status="${RED}Not available${NC}"
    fi

    echo -e "  [1] Python (Native)     - Fastest startup           [$python_status]"
    echo -e "  [2] Docker (Standard)   - No Python setup needed    [$docker_status]"
    echo -e "  [3] Docker (AI)         - With AI validation        [$ai_status]"
    echo ""
    echo "  [0] Cancel"
    echo ""

    while true; do
        read -p "Select mode [1-3]: " choice
        case $choice in
            1)
                if check_python && check_python_deps; then
                    MODE="python"
                    return 0
                else
                    echo -e "${RED}Python mode not available. Install dependencies first.${NC}"
                fi
                ;;
            2)
                if check_docker; then
                    MODE="docker"
                    return 0
                else
                    echo -e "${RED}Docker not available. Please install Docker.${NC}"
                fi
                ;;
            3)
                if check_docker; then
                    MODE="ai"
                    return 0
                else
                    echo -e "${RED}Docker not available. Please install Docker.${NC}"
                fi
                ;;
            0|q|Q)
                echo "Cancelled."
                exit 0
                ;;
            *)
                echo "Please enter 1, 2, 3, or 0 to cancel"
                ;;
        esac
    done
}

# Function to run Python mode
run_python() {
    local image="$1"
    shift
    local args="$@"

    echo -e "${YELLOW}Running with Python...${NC}"
    echo ""

    cd "$PROJECT_ROOT"
    $PYTHON_CMD src/extract_km.py "$image" $args
}

# Function to run Docker mode
run_docker() {
    local image="$1"
    shift
    local args="$@"

    echo -e "${YELLOW}Running with Docker...${NC}"
    echo ""

    cd "$PROJECT_ROOT"
    bash docker/run.sh "$image" $args
}

# Function to run AI mode
run_ai() {
    local image="$1"
    shift
    local args="$@"

    echo -e "${YELLOW}Running with AI validation...${NC}"
    echo ""

    # Add --validate flag if not already present
    if [[ ! "$args" =~ "--validate" ]]; then
        args="$args --validate"
    fi

    cd "$PROJECT_ROOT"
    bash docker/run-ai.sh "$image" $args
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        --status)
            show_status
            exit 0
            ;;
        --start)
            # Start AI services
            cd "$PROJECT_ROOT"
            bash docker/run-ai.sh --start
            exit 0
            ;;
        --stop)
            # Stop AI services
            cd "$PROJECT_ROOT"
            bash docker/run-ai.sh --stop
            exit 0
            ;;
        -*)
            # Collect other flags as extra args
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
        *)
            # First non-flag argument is the image file
            if [ -z "$IMAGE_FILE" ]; then
                IMAGE_FILE="$1"
            else
                EXTRA_ARGS="$EXTRA_ARGS $1"
            fi
            shift
            ;;
    esac
done

# Show banner
show_banner

# Check if image file provided
if [ -z "$IMAGE_FILE" ]; then
    echo -e "${YELLOW}No image file specified.${NC}"
    echo ""
    echo "Usage: $0 [--mode python|docker|ai] <image_file> [options]"
    echo ""
    echo "Run '$0 --help' for more information."
    echo "Run '$0 --status' to check system status."
    exit 1
fi

# Check if image file exists
if [ ! -f "$IMAGE_FILE" ]; then
    echo -e "${RED}Error: Image file not found: $IMAGE_FILE${NC}"
    exit 1
fi

# If mode not specified, ask interactively
if [ -z "$MODE" ]; then
    select_mode_interactive
fi

# Validate mode
case $MODE in
    python|py|native)
        MODE="python"
        if ! check_python; then
            echo -e "${RED}Error: Python not found${NC}"
            exit 1
        fi
        if ! check_python_deps; then
            echo -e "${RED}Error: Python dependencies not installed${NC}"
            echo "Run: pip install -r requirements.txt"
            exit 1
        fi
        ;;
    docker|container)
        MODE="docker"
        if ! check_docker; then
            echo -e "${RED}Error: Docker not available${NC}"
            exit 1
        fi
        ;;
    ai|ai-docker|validate)
        MODE="ai"
        if ! check_docker; then
            echo -e "${RED}Error: Docker not available (required for AI mode)${NC}"
            exit 1
        fi
        ;;
    *)
        echo -e "${RED}Error: Unknown mode '$MODE'${NC}"
        echo "Valid modes: python, docker, ai"
        exit 1
        ;;
esac

echo -e "Mode: ${BOLD}$MODE${NC}"
echo -e "Image: ${BOLD}$IMAGE_FILE${NC}"
if [ -n "$EXTRA_ARGS" ]; then
    echo -e "Options: ${BOLD}$EXTRA_ARGS${NC}"
fi
echo ""

# Run selected mode
case $MODE in
    python)
        run_python "$IMAGE_FILE" $EXTRA_ARGS
        ;;
    docker)
        run_docker "$IMAGE_FILE" $EXTRA_ARGS
        ;;
    ai)
        run_ai "$IMAGE_FILE" $EXTRA_ARGS
        ;;
esac

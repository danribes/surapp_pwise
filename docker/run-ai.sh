#!/bin/bash
# SURAPP AI-Enhanced Docker Runner
#
# This script runs SURAPP with AI-powered validation using Ollama.
#
# Usage:
#   ./run-ai.sh <image_file> [options]
#
# Examples:
#   ./run-ai.sh my_plot.png --validate
#   ./run-ai.sh my_plot.png --time-max 24 --validate
#   ./run-ai.sh --status  # Check AI service status

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    echo "Please install Docker from https://www.docker.com/get-started"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed.${NC}"
    echo "Please install Docker Compose or update Docker Desktop"
    exit 1
fi

# Use 'docker compose' if available, otherwise 'docker-compose'
if docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Function to start AI services
start_ai_services() {
    echo -e "${YELLOW}Starting AI services...${NC}"
    cd "$SCRIPT_DIR"
    $COMPOSE_CMD -f docker-compose.yml -f docker-compose.ai.yml up -d ollama

    echo -e "${YELLOW}Waiting for Ollama to be ready...${NC}"
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo -e "${GREEN}Ollama is ready!${NC}"
            return 0
        fi
        sleep 2
        echo -n "."
    done

    echo -e "\n${RED}Timeout waiting for Ollama${NC}"
    return 1
}

# Function to ensure model is available
ensure_model() {
    local model="llama3.2-vision"
    echo -e "${YELLOW}Checking for $model model...${NC}"

    # Check if model exists
    if curl -s http://localhost:11434/api/tags | grep -q "$model"; then
        echo -e "${GREEN}Model $model is available${NC}"
        return 0
    fi

    echo -e "${YELLOW}Pulling $model (this may take a while, ~4GB)...${NC}"
    docker exec surapp-ollama ollama pull $model
    echo -e "${GREEN}Model ready!${NC}"
}

# Function to show status
show_status() {
    echo -e "${CYAN}SURAPP AI Service Status${NC}"
    echo "========================="

    # Check if Ollama container is running
    if docker ps --format '{{.Names}}' | grep -q "surapp-ollama"; then
        echo -e "Ollama Container: ${GREEN}Running${NC}"

        # Check API
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo -e "Ollama API: ${GREEN}Available${NC}"

            # List models
            echo ""
            echo "Installed Models:"
            curl -s http://localhost:11434/api/tags | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = data.get('models', [])
if models:
    for m in models:
        print(f\"  - {m.get('name', 'unknown')}\")
else:
    print('  (none)')
"
        else
            echo -e "Ollama API: ${RED}Not responding${NC}"
        fi
    else
        echo -e "Ollama Container: ${RED}Not running${NC}"
        echo ""
        echo "Start with:"
        echo "  ./run-ai.sh --start"
    fi
}

# Parse special commands
case "${1:-}" in
    --status)
        show_status
        exit 0
        ;;
    --start)
        start_ai_services
        ensure_model
        echo ""
        show_status
        exit 0
        ;;
    --stop)
        echo -e "${YELLOW}Stopping AI services...${NC}"
        cd "$SCRIPT_DIR"
        $COMPOSE_CMD -f docker-compose.yml -f docker-compose.ai.yml down
        echo -e "${GREEN}AI services stopped${NC}"
        exit 0
        ;;
    --help|-h|"")
        echo -e "${CYAN}SURAPP AI-Enhanced - Kaplan-Meier Curve Extractor${NC}"
        echo ""
        echo "Usage: $0 <image_file> [options]"
        echo "       $0 --status    # Check AI service status"
        echo "       $0 --start     # Start AI services"
        echo "       $0 --stop      # Stop AI services"
        echo ""
        echo "Options:"
        echo "  --time-max TIME    Maximum time value on X-axis"
        echo "  --curves N         Expected number of curves (default: 2)"
        echo "  --validate         Enable AI validation"
        echo "  -o, --output DIR   Output directory"
        echo ""
        echo "Examples:"
        echo "  $0 --start                              # Start AI services first"
        echo "  $0 my_km_plot.png --validate            # Extract with AI validation"
        echo "  $0 my_km_plot.png --time-max 24 --validate"
        exit 0
        ;;
esac

IMAGE_FILE="$1"
shift  # Remove first argument, keep the rest as options

# Check if image file exists
if [ ! -f "$IMAGE_FILE" ]; then
    echo -e "${RED}Error: Image file not found: $IMAGE_FILE${NC}"
    exit 1
fi

# Ensure AI services are running
if ! docker ps --format '{{.Names}}' | grep -q "surapp-ollama"; then
    echo -e "${YELLOW}AI services not running. Starting...${NC}"
    start_ai_services
    ensure_model
    echo ""
fi

# Get absolute path and filename
IMAGE_DIR=$(cd "$(dirname "$IMAGE_FILE")" && pwd)
IMAGE_NAME=$(basename "$IMAGE_FILE")

# Create output directory
OUTPUT_DIR="${IMAGE_DIR}/results"
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}SURAPP AI-Enhanced - Kaplan-Meier Curve Extractor${NC}"
echo "=================================================="
echo "Input:  $IMAGE_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

# Build AI image if not exists
if ! docker image inspect surapp-ai:latest &> /dev/null; then
    echo -e "${YELLOW}Building AI-enhanced Docker image (first run only)...${NC}"
    cd "$SCRIPT_DIR"
    $COMPOSE_CMD -f docker-compose.yml -f docker-compose.ai.yml build surapp-ai
    echo ""
fi

# Run the AI-enhanced extraction
echo -e "${YELLOW}Running AI-enhanced extraction...${NC}"
echo ""

docker run --rm \
    --network host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -e AI_MODEL=llama3.2-vision \
    -e AI_ENABLED=true \
    -v "$IMAGE_DIR:/data/input:ro" \
    -v "$OUTPUT_DIR:/data/output" \
    surapp-ai:latest \
    python /app/extract_km_ai.py "/data/input/$IMAGE_NAME" -o "/data/output" "$@"

echo ""
echo -e "${GREEN}Done! Results saved to: $OUTPUT_DIR${NC}"

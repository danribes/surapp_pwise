#!/bin/bash
# SURAPP Docker Runner
#
# This script simplifies running SURAPP in Docker.
#
# Usage:
#   ./docker-run.sh <image_file> [options]
#
# Examples:
#   ./docker-run.sh my_plot.png
#   ./docker-run.sh my_plot.png --time-max 24
#   ./docker-run.sh my_plot.png --curves 3 --time-max 36

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    echo "Please install Docker from https://www.docker.com/get-started"
    exit 1
fi

# Check if image argument provided
if [ $# -lt 1 ]; then
    echo -e "${YELLOW}SURAPP - Kaplan-Meier Curve Extractor${NC}"
    echo ""
    echo "Usage: $0 <image_file> [options]"
    echo ""
    echo "Options:"
    echo "  --time-max TIME    Maximum time value on X-axis"
    echo "  --curves N         Expected number of curves (default: 2)"
    echo "  -o, --output DIR   Output directory"
    echo ""
    echo "Examples:"
    echo "  $0 my_km_plot.png"
    echo "  $0 my_km_plot.png --time-max 24"
    echo "  $0 my_km_plot.png --curves 2 --time-max 36"
    exit 0
fi

IMAGE_FILE="$1"
shift  # Remove first argument, keep the rest as options

# Check if image file exists
if [ ! -f "$IMAGE_FILE" ]; then
    echo -e "${RED}Error: Image file not found: $IMAGE_FILE${NC}"
    exit 1
fi

# Get absolute path and filename
IMAGE_DIR=$(cd "$(dirname "$IMAGE_FILE")" && pwd)
IMAGE_NAME=$(basename "$IMAGE_FILE")

# Create output directory if it doesn't exist
OUTPUT_DIR="${IMAGE_DIR}/results"
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}SURAPP - Kaplan-Meier Curve Extractor${NC}"
echo "========================================"
echo "Input:  $IMAGE_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

# Build Docker image if not exists
if ! docker image inspect surapp:latest &> /dev/null; then
    echo -e "${YELLOW}Building Docker image (first run only)...${NC}"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    docker build -t surapp:latest "$SCRIPT_DIR"
    echo ""
fi

# Run the extraction
echo -e "${YELLOW}Running extraction...${NC}"
echo ""

docker run --rm \
    -v "$IMAGE_DIR:/data/input:ro" \
    -v "$OUTPUT_DIR:/data/output" \
    surapp:latest \
    python /app/extract_km.py "/data/input/$IMAGE_NAME" -o "/data/output" "$@"

echo ""
echo -e "${GREEN}Done! Results saved to: $OUTPUT_DIR${NC}"

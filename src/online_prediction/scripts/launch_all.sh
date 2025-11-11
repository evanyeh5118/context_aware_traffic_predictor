#!/bin/bash

# UDP CSV Data Relay System - Launcher Script
# This script launches the receiver, relay, and sender together

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Path to Python components
PY_DIR="${SCRIPT_DIR}/../Networks"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if required Python files exist
for file in receiver.py relay.py relay_predictor.py sender.py; do
    if [ ! -f "$PY_DIR/$file" ]; then
        echo "Error: $file not found in $PY_DIR"
        exit 1
    fi
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}UDP CSV Data Relay System${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Initialize environment for sender (values can be overridden if already set)
: "${RELAY_IP:=127.0.0.1}"
: "${RELAY_PORT:=5000}"
: "${REPLAY_REAL_TIMING:=1}"
: "${TIME_SCALE:=1.0}"
: "${VERBOSE:=0}"
: "${USE_PREDICTOR:=0}"

# Resolve default CSV_FILE if not provided
if [ -z "${CSV_FILE}" ]; then
    DEFAULT_CSV_REL="${SCRIPT_DIR}/../../../data/processed/dpdr/combined_flows_forward_20.csv"
    if command -v realpath >/dev/null 2>&1; then    
        CSV_FILE="$(realpath "${DEFAULT_CSV_REL}")"
    else
        # Fallback without realpath
        pushd "${SCRIPT_DIR}" >/dev/null
        CSV_FILE="$(pwd)/../../../data/processed/dpdr/combined_flows_forward_20.csv"
        popd >/dev/null
    fi
fi

export RELAY_IP RELAY_PORT REPLAY_REAL_TIMING TIME_SCALE VERBOSE CSV_FILE USE_PREDICTOR

echo "Configuration:"
echo "  CSV_FILE=${CSV_FILE}"
echo "  RELAY_IP=${RELAY_IP}  RELAY_PORT=${RELAY_PORT}"
echo "  REPLAY_REAL_TIMING=${REPLAY_REAL_TIMING}  TIME_SCALE=${TIME_SCALE}"
echo "  VERBOSE=${VERBOSE}"
echo "  USE_PREDICTOR=${USE_PREDICTOR} (0=relay.py, 1=relay_predictor.py)"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down all processes...${NC}"
    
    # Kill all background processes
    kill $(jobs -p) 2>/dev/null || true
    
    wait 2>/dev/null || true
    
    echo -e "${YELLOW}All processes stopped.${NC}"
}

# Set trap to call cleanup on script exit
trap cleanup EXIT INT TERM

# Start the receiver in the background
echo -e "${GREEN}[1/3]${NC} Starting Receiver..."
python3 "$PY_DIR/receiver.py" &
RECEIVER_PID=$!
echo -e "${GREEN}✓ Receiver started (PID: $RECEIVER_PID)${NC}"

# Give receiver time to start
sleep 1

# Start the relay in the background
echo -e "${GREEN}[2/3]${NC} Starting Relay..."
if [ "$USE_PREDICTOR" = "1" ]; then
    echo "  Using relay_predictor.py (with ML predictions)"
    python3 "$PY_DIR/relay_predictor.py" &
else
    echo "  Using relay.py (standard relay)"
    python3 "$PY_DIR/relay.py" &
fi
RELAY_PID=$!
echo -e "${GREEN}✓ Relay started (PID: $RELAY_PID)${NC}"

# Give relay time to start
sleep 1

# Start the sender in the background
echo -e "${GREEN}[3/3]${NC} Starting Sender..."
python3 "$PY_DIR/sender.py" &
SENDER_PID=$!
echo -e "${GREEN}✓ Sender started (PID: $SENDER_PID)${NC}"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All components are running!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Process IDs:"
echo "  Receiver: $RECEIVER_PID"
echo "  Relay:    $RELAY_PID"
echo "  Sender:   $SENDER_PID"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all processes${NC}"
echo ""

# Wait for all background processes
wait

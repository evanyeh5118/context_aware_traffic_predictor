#!/bin/bash

# UDP CSV Data Relay System - Launcher Script
# This script launches the receiver, relay, and sender together

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Path to Python components
PY_DIR="${SCRIPT_DIR}/../Networks"

# Path to config file
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config.yaml not found in $SCRIPT_DIR"
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

# Function to read YAML config using Python
read_config() {
    python3 -c "
import yaml
import sys
import os

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract values
    relay_ip = config['relay']['ip']
    relay_port = config['relay']['port']
    replay_real_timing = config['replay']['real_timing']
    time_scale = config['replay']['time_scale']
    csv_file = config['data']['csv_file']
    verbose = config['settings']['verbose']
    use_predictor = config['settings']['use_predictor']
    
    # Extract predictor configuration (optional)
    config_path = config.get('predictor', {}).get('config_path', '')
    model_folder_context_aware = config.get('predictor', {}).get('model_folder_context_aware', '')
    model_folder_context_free = config.get('predictor', {}).get('model_folder_context_free', '')
    
    # Resolve CSV file path (relative to script directory)
    if not os.path.isabs(csv_file):
        csv_file = os.path.abspath(os.path.join('$SCRIPT_DIR', csv_file))
    
    # Resolve predictor paths (relative to Networks directory)
    py_dir = '$PY_DIR'
    if config_path and not os.path.isabs(config_path):
        config_path = os.path.abspath(os.path.join(py_dir, config_path))
    if model_folder_context_aware and not os.path.isabs(model_folder_context_aware):
        model_folder_context_aware = os.path.abspath(os.path.join(py_dir, model_folder_context_aware))
    if model_folder_context_free and not os.path.isabs(model_folder_context_free):
        model_folder_context_free = os.path.abspath(os.path.join(py_dir, model_folder_context_free))
    
    # Output as shell variable assignments
    print(f'RELAY_IP=\"{relay_ip}\"')
    print(f'RELAY_PORT=\"{relay_port}\"')
    print(f'REPLAY_REAL_TIMING=\"{replay_real_timing}\"')
    print(f'TIME_SCALE=\"{time_scale}\"')
    print(f'CSV_FILE=\"{csv_file}\"')
    print(f'VERBOSE=\"{verbose}\"')
    print(f'USE_PREDICTOR=\"{use_predictor}\"')
    print(f'CONFIG_PATH=\"{config_path}\"')
    print(f'MODEL_FOLDER_CONTEXT_AWARE=\"{model_folder_context_aware}\"')
    print(f'MODEL_FOLDER_CONTEXT_FREE=\"{model_folder_context_free}\"')
    
except FileNotFoundError:
    print('Error: config.yaml not found', file=sys.stderr)
    sys.exit(1)
except yaml.YAMLError as e:
    print(f'Error parsing YAML: {e}', file=sys.stderr)
    sys.exit(1)
except KeyError as e:
    print(f'Error: Missing key in config: {e}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'Error reading config: {e}', file=sys.stderr)
    sys.exit(1)
"
}

# Load configuration from YAML (unless already set via environment variables)
if [ -z "${RELAY_IP}" ]; then
    echo "Loading configuration from config.yaml..."
    eval "$(read_config)"
else
    echo "Using environment variables (config.yaml overridden)..."
    # Still load defaults for any missing variables
    eval "$(read_config)" 2>/dev/null || true
fi

export RELAY_IP RELAY_PORT REPLAY_REAL_TIMING TIME_SCALE VERBOSE CSV_FILE USE_PREDICTOR
export CONFIG_PATH MODEL_FOLDER_CONTEXT_AWARE MODEL_FOLDER_CONTEXT_FREE

echo "Configuration:"
echo "  CSV_FILE=${CSV_FILE}"
echo "  RELAY_IP=${RELAY_IP}  RELAY_PORT=${RELAY_PORT}"
echo "  REPLAY_REAL_TIMING=${REPLAY_REAL_TIMING}  TIME_SCALE=${TIME_SCALE}"
echo "  VERBOSE=${VERBOSE}"
echo "  USE_PREDICTOR=${USE_PREDICTOR} (0=relay.py, 1=relay_predictor.py)"
if [ "$USE_PREDICTOR" = "1" ]; then
    echo ""
    echo "Predictor Configuration:"
    echo "  CONFIG_PATH=${CONFIG_PATH}"
    echo "  MODEL_FOLDER_CONTEXT_AWARE=${MODEL_FOLDER_CONTEXT_AWARE}"
    echo "  MODEL_FOLDER_CONTEXT_FREE=${MODEL_FOLDER_CONTEXT_FREE}"
fi
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

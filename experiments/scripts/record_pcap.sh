#!/usr/bin/env bash

set -euo pipefail

# Record UDP traffic while running launch_all.sh using tcpdump

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check dependencies
if ! command -v tcpdump >/dev/null 2>&1; then
  echo "Error: tcpdump is not installed or not in PATH" >&2
  exit 1
fi

# Defaults (can be overridden via env)
: "${INTERFACE:=}"
: "${BPF_FILTER:=udp port 5000 or udp port 5001}"
: "${OUTPUT_DIR:=${SCRIPT_DIR}../../../data/captures}"
: "${DURATION:=10}"

# Auto-detect interface if not provided
# Since all components bind to 127.0.0.1 or 0.0.0.0, traffic is on loopback interface
if [ -z "${INTERFACE}" ]; then
  # First, try to detect loopback interface (priority since traffic is on 127.0.0.1)
  if command -v tcpdump >/dev/null 2>&1; then
    # Get list of interfaces from tcpdump
    INTERFACE_LIST=$(tcpdump -D 2>/dev/null || true)
    
    if [ -n "${INTERFACE_LIST}" ]; then
      # Try common loopback interface names/patterns
      LOOPBACK_INTERFACE=""
      
      # Look for interfaces containing "loop" or "lo" (case insensitive)
      LOOPBACK_LINE=$(echo "${INTERFACE_LIST}" | grep -i "loop\|^[0-9]*\.\s*lo\b" | head -1)
      
      if [ -n "${LOOPBACK_LINE}" ]; then
        # Extract interface name/number - could be number or name
        # Format: "1. lo" or "1. Loopback" or "1. Loopback Pseudo-Interface 1"
        LOOPBACK_INTERFACE=$(echo "${LOOPBACK_LINE}" | sed -E 's/^[0-9]+\.\s*//' | awk '{print $1}')
        
        # If extraction failed, try using the number from the line
        if [ -z "${LOOPBACK_INTERFACE}" ] || [ "${LOOPBACK_INTERFACE}" = "${LOOPBACK_LINE}" ]; then
          LOOPBACK_INTERFACE=$(echo "${LOOPBACK_LINE}" | sed -E 's/\..*$//')
        fi
        
        if [ -n "${LOOPBACK_INTERFACE}" ]; then
          INTERFACE="${LOOPBACK_INTERFACE}"
          echo "Detected loopback interface: ${INTERFACE}"
        fi
      fi
    fi
  fi
  
  # If still not found, try Linux-style detection
  if [ -z "${INTERFACE}" ] && command -v ip >/dev/null 2>&1; then
    # Best effort: pick interface used for default route
    INTERFACE="$(ip route get 1.1.1.1 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}')"
  fi
  
  # Final fallback: default loopback names
  if [ -z "${INTERFACE}" ]; then
    echo "Warning: Could not auto-detect loopback interface. Attempting default 'lo'." >&2
    echo "Available interfaces:" >&2
    tcpdump -D 2>/dev/null | head -10 || echo "  (run 'tcpdump -D' to see all interfaces)" >&2
    INTERFACE="lo"
  fi
fi

mkdir -p "${OUTPUT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
PCAP_FILE="${OUTPUT_DIR}/traffic_${TS}.pcap"

echo "========================================"
echo "Recording PCAP while launching the system"
echo "========================================"
echo "Interface: ${INTERFACE}"
echo "PCAP file: ${PCAP_FILE}"
echo "BPF filter: ${BPF_FILTER}"
if [ -n "${DURATION}" ]; then
  echo "Duration: ${DURATION}s"
fi
echo ""
echo "Note: All components use loopback (127.0.0.1), so interface should be loopback."
echo "Expected traffic flow:"
echo "  Sender -> Relay (port 5000) -> Receiver (port 5001)"
echo "  (or Sender -> Receiver directly if RELAY_PORT=5001)"
echo ""

# Cleanup handler
TCPDUMP_PID=""
LAUNCH_PID=""
cleanup() {
  echo "\nStopping capture and launched processes..."
  if [ -n "${LAUNCH_PID}" ] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill "${LAUNCH_PID}" 2>/dev/null || true
    wait "${LAUNCH_PID}" 2>/dev/null || true
  fi
  if [ -n "${TCPDUMP_PID}" ] && kill -0 "${TCPDUMP_PID}" 2>/dev/null; then
    kill "${TCPDUMP_PID}" 2>/dev/null || true
    wait "${TCPDUMP_PID}" 2>/dev/null || true
  fi
  echo "Done. PCAP saved to: ${PCAP_FILE}"
}
trap cleanup EXIT INT TERM

# Start tcpdump (may require sudo depending on system permissions)
echo "Starting tcpdump..."
if [ -n "${DURATION}" ]; then
  # Use timeout to stop after DURATION seconds
  # Quote BPF_FILTER to handle spaces correctly
  timeout "${DURATION}"s tcpdump -i "${INTERFACE}" -w "${PCAP_FILE}" "${BPF_FILTER}" &
else
  # Quote BPF_FILTER to handle spaces correctly
  tcpdump -i "${INTERFACE}" -w "${PCAP_FILE}" "${BPF_FILTER}" &
fi
TCPDUMP_PID=$!
echo "tcpdump started (PID: ${TCPDUMP_PID})"
# Give tcpdump a moment to start
sleep 0.5
# Check if tcpdump is still running (might have failed silently)
if ! kill -0 "${TCPDUMP_PID}" 2>/dev/null; then
  echo "Error: tcpdump failed to start. Check permissions (may need sudo) and interface name." >&2
  exit 1
fi

# Launch the system
echo "Starting launch_all.sh..."
"${SCRIPT_DIR}/launch_all.sh" &
LAUNCH_PID=$!
echo "launch_all.sh started (PID: ${LAUNCH_PID})"

# If DURATION is set, wait until timeout finishes then stop launcher
if [ -n "${DURATION}" ]; then
  wait "${TCPDUMP_PID}" 2>/dev/null || true
  # Stop launcher after capture finishes
  if kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    echo "Duration elapsed, stopping launched processes..."
    kill "${LAUNCH_PID}" 2>/dev/null || true
    wait "${LAUNCH_PID}" 2>/dev/null || true
  fi
else
  # Otherwise, just wait for launcher; user can Ctrl+C to stop
  wait "${LAUNCH_PID}" 2>/dev/null || true
fi

echo "All done."
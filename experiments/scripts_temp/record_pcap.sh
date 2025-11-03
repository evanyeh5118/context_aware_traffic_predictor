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
if [ -z "${INTERFACE}" ]; then
  if command -v ip >/dev/null 2>&1; then
    # Best effort: pick interface used for default route
    INTERFACE="$(ip route get 1.1.1.1 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}')"
  fi
  if [ -z "${INTERFACE}" ]; then
    # Fallback to loopback
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
  timeout "${DURATION}"s tcpdump -i "${INTERFACE}" -w "${PCAP_FILE}" ${BPF_FILTER} &
else
  tcpdump -i "${INTERFACE}" -w "${PCAP_FILE}" ${BPF_FILTER} &
fi
TCPDUMP_PID=$!
echo "tcpdump started (PID: ${TCPDUMP_PID})"

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



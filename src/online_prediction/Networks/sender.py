import socket
import csv
import time
import sys
import signal
import os

# Configuration
# Default to DPDR forward trajectory CSV; can be overridden via env or argv
CSV_FILE = os.environ.get(
    "CSV_FILE",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "processed",
        "dpdr",
        "combined_flows_forward.csv",
    ),
)
RELAY_IP = os.environ.get("RELAY_IP", "127.0.0.1")
# Align with receiver default port (5001)
RELAY_PORT = int(os.environ.get("RELAY_PORT", "5001"))

# Replay timing: sleep based on Time deltas (scaled)
REPLAY_REAL_TIMING = os.environ.get("REPLAY_REAL_TIMING", "1") == "1"
TIME_SCALE = float(os.environ.get("TIME_SCALE", "1.0"))  # 1.0 = real-time
VERBOSE = os.environ.get("VERBOSE", "1") == "1"

# Global flag for graceful shutdown
shutdown_flag = False


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    shutdown_flag = True
    print("\nShutdown signal received")


def vprint(message: str) -> None:
    if VERBOSE:
        print(message)


def _parse_header(header):
    """Return indices for time and transmit flag, plus a list of payload column indices."""
    header_to_index = {name.strip(): idx for idx, name in enumerate(header)}

    # Common DPDR headers
    time_idx = header_to_index.get("Time")
    flag_idx = header_to_index.get("Transmition Flags")

    # All other columns are payload
    payload_indices = [i for i in range(len(header)) if i not in {time_idx, flag_idx}]

    return time_idx, flag_idx, payload_indices


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _load_all_rows(csv_path):
    """Load entire CSV and return (header, time_idx, flag_idx, payload_indices, rows)."""
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            print("Error: CSV has no header")
            sys.exit(1)

        time_idx, flag_idx, payload_indices = _parse_header(header)
        vprint(f"CSV Header: {','.join(header)}")
        vprint(
            f"Resolved columns -> time_idx={time_idx}, flag_idx={flag_idx}, payload_cols={payload_indices}"
        )

        rows = [row for row in reader if row]
        return header, time_idx, flag_idx, payload_indices, rows


def send_csv_rows():
    global shutdown_flag
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        header, time_idx, flag_idx, payload_indices, rows = _load_all_rows(CSV_FILE)

        # Establish real-time baseline for scheduling
        sent_count = 0
        total_rows = len(rows)

        # Determine base CSV time (first valid time) for relative scheduling
        base_csv_time = None
        if time_idx is not None:
            for r in rows:
                if time_idx < len(r):
                    t = _safe_float(r[time_idx], None)
                    if t is not None:
                        base_csv_time = t
                        break

        start_wall = time.perf_counter()

        for i, row in enumerate(rows, start=1):
            if shutdown_flag:
                break

            # Determine whether to transmit based on flag
            should_transmit = True
            if flag_idx is not None and 0 <= flag_idx < len(row):
                should_transmit = _safe_float(row[flag_idx], 0.0) > 0.0

            # Schedule based on real-time counter and CSV 'Time'
            if REPLAY_REAL_TIMING and time_idx is not None and 0 <= time_idx < len(row) and base_csv_time is not None:
                row_time = _safe_float(row[time_idx], base_csv_time)
                relative_seconds = max(0.0, (row_time - base_csv_time) / max(1e-9, TIME_SCALE))
                target_wall = start_wall + relative_seconds
                now = time.perf_counter()
                sleep_secs = target_wall - now
                if sleep_secs > 0:
                    time.sleep(sleep_secs)

            if not should_transmit:
                continue

            message = ",".join(row)
            sock.sendto(message.encode(), (RELAY_IP, RELAY_PORT))
            sent_count += 1
            vprint(f"Sent {sent_count} (row {i}/{total_rows}): {message}")

        if not shutdown_flag:
            print(f"\nFinished. Sent {sent_count} messages out of {total_rows} rows")
        else:
            print(f"\nStopped. Sent {sent_count} messages out of {total_rows} rows")

    except FileNotFoundError:
        print(f"Error: CSV file '{CSV_FILE}' not found")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\nSender stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    print("Starting UDP sender...")
    print(f"Target: {RELAY_IP}:{RELAY_PORT}")
    print(f"CSV: {CSV_FILE}")
    if REPLAY_REAL_TIMING:
        print(f"Timing: replaying 'Time' deltas (scale={TIME_SCALE})\n")
    else:
        print("Timing: no replay (sending as fast as possible)\n")
    send_csv_rows()
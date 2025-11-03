import socket
import signal
import sys

# Configuration
LISTEN_IP = '0.0.0.0'
LISTEN_PORT = 5001
BUFFER_SIZE = 4096

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    shutdown_flag = True
    print("\nShutdown signal received")

def receive_packets():
    global shutdown_flag
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_IP, LISTEN_PORT))
    
    print(f"Receiver listening on {LISTEN_IP}:{LISTEN_PORT}\n")
    
    try:
        msg_count = 0
        while not shutdown_flag:
            # Set socket timeout to allow checking shutdown_flag periodically
            sock.settimeout(1.0)
            try:
                data, addr = sock.recvfrom(BUFFER_SIZE)
                msg_count += 1
                payload = data.decode()
                src_ip, src_port = addr
                
                print(f"Message {msg_count} received from {src_ip}:{src_port}")
                print(f"Data: {payload}\n")
            except socket.timeout:
                continue
    
    except KeyboardInterrupt:
        print(f"\nReceiver stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()
        print(f"Receiver stopped. Total messages: {msg_count}")

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    receive_packets()
import socket
import signal
import sys

# Configuration
LISTEN_IP = '0.0.0.0'
LISTEN_PORT = 5000
RECEIVER_IP = '127.0.0.1'
RECEIVER_PORT = 5001
BUFFER_SIZE = 4096

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    shutdown_flag = True
    print("\nShutdown signal received")

def relay_packets():
    global shutdown_flag
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_IP, LISTEN_PORT))
    
    print(f"Relay listening on {LISTEN_IP}:{LISTEN_PORT}")
    print(f"Forwarding to {RECEIVER_IP}:{RECEIVER_PORT}\n")
    
    try:
        while not shutdown_flag:
            # Set socket timeout to allow checking shutdown_flag periodically
            sock.settimeout(1.0)
            try:
                data, addr = sock.recvfrom(BUFFER_SIZE)
                payload = data.decode()
                src_ip, src_port = addr
                
                # Print received packet info
                print(f"--- Packet Received ---")
                print(f"Source IP: {src_ip}:{src_port}")
                print(f"Destination IP: {LISTEN_IP}:{LISTEN_PORT}")
                print(f"Payload: {payload}")
                print(f"--- Forwarding ---\n")
                
                # Forward to receiver
                sock.sendto(data, (RECEIVER_IP, RECEIVER_PORT))
            except socket.timeout:
                continue
    
    except KeyboardInterrupt:
        print("\nRelay stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()
        print("Relay stopped")

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    relay_packets()
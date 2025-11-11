import socket
import signal
import json
import pickle
import sys
import os
# Add the project's root directory to sys.path, so that 'src' can be imported as a module
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.online_prediction import OnlinePredictor
from src.context_aware.models import createModel

configPath = "../../../experiments/config/combined_flows_forward.json"
modelFolder = "../../../data/models/context_aware"

class RelayPredictor:
    """UDP packet relay that forwards packets from one port to another."""
    
    def __init__(self, listen_ip='0.0.0.0', listen_port=5000, 
                 receiver_ip='127.0.0.1', receiver_port=5001, 
                 buffer_size=4096):
        """Initialize the relay with configuration parameters.
        
        Args:
            listen_ip: IP address to listen on (default: '0.0.0.0')
            listen_port: Port to listen on (default: 5000)
            receiver_ip: IP address to forward packets to (default: '127.0.0.1')
            receiver_port: Port to forward packets to (default: 5001)
            buffer_size: Buffer size for receiving packets (default: 4096)
        """
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.receiver_ip = receiver_ip
        self.receiver_port = receiver_port
        self.buffer_size = buffer_size
        
        self.shutdown_flag = False
        self.sock = None
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        self._initializePredictor()

    def _initializePredictor(self):
        config = json.load(open(configPath))
        name = config.get("NAME")
        len_window = config.get("LEN_WINDOW")

        with open(f"{modelFolder}/{name}_lenWindow_{len_window}_modelConfig.pkl", "rb") as f:
            modelConfig = pickle.load(f)
        with open(f"{modelFolder}/{name}_lenWindow_{len_window}_metaConfig.pkl", "rb") as f:
            metaConfig = pickle.load(f)
        metaConfig.display()

        model, _ = createModel(modelConfig)
        model.load_checkpoint(f"{modelFolder}/{name}_lenWindow_{len_window}.pth")
        
        self.onlinePredictor = OnlinePredictor(model, metaConfig)
    
    def signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully"""
        self.shutdown_flag = True
        print("\nShutdown signal received")
    
    def run(self):
        """Start the relay service."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.listen_ip, self.listen_port))
        
        print(f"Relay listening on {self.listen_ip}:{self.listen_port}")
        print(f"Forwarding to {self.receiver_ip}:{self.receiver_port}\n")
        
        try:
            while not self.shutdown_flag:
                # Set socket timeout to allow checking shutdown_flag periodically
                self.sock.settimeout(1.0)
                try:
                    data, addr = self.sock.recvfrom(self.buffer_size)
                    payload = data.decode()
                    src_ip, src_port = addr
                    
                    # Print received packet info
                    print(f"--- Packet Received ---")
                    print(f"Source IP: {src_ip}:{src_port}")
                    print(f"Destination IP: {self.listen_ip}:{self.listen_port}")
                    print(f"Payload: {payload}")
                    print(f"--- Forwarding ---\n")
                    
                    # Forward to receiver
                    self.sock.sendto(data, (self.receiver_ip, self.receiver_port))
                except socket.timeout:
                    continue
        
        except KeyboardInterrupt:
            print("\nRelay stopped by user")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the relay and close the socket."""
        if self.sock:
            self.sock.close()
            self.sock = None
        print("Relay stopped")


if __name__ == "__main__":
    relay = RelayPredictor()
    relay.run()
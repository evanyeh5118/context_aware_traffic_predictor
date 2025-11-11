import socket
import signal
import json
import pickle
import sys
import os
import csv
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Add the project's root directory to sys.path, so that 'src' can be imported as a module
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.online_prediction import OnlinePredictor
from src.context_aware.models import createModel

configPath = "../../../experiments/config/combined_flows_forward_20.json"
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
        
        self.onlinePredictor = None
        self.last_prediction = None
        self.traffic_recieved_list = []
        self.traffic_predicted_list = []
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        self._initializePredictor()

    def _initializePredictor(self):
        config = json.load(open(configPath))
        name = config.get("NAME")
        len_window = config.get("LEN_WINDOW")

        with open(f"{modelFolder}/{name}_modelConfig.pkl", "rb") as f:
            modelConfig = pickle.load(f)
        with open(f"{modelFolder}/{name}_metaConfig.pkl", "rb") as f:
            metaConfig = pickle.load(f)
        metaConfig.display()

        model, _ = createModel(modelConfig)
        model.load_checkpoint(f"{modelFolder}/{name}.pth")
        
        self.onlinePredictor = OnlinePredictor(model, metaConfig)
    
    def signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully"""
        self.shutdown_flag = True
        print("\nShutdown signal received")

    def _onlinePredictor(self, payload):
        payload = payload.split(",")
        payload = [float(x) for x in payload]
        self.onlinePredictor.receive(payload)
        traffic_predicted, traffic_recieved = self.onlinePredictor.predict()
        traffic_predicted = np.round(traffic_predicted, 0).astype(int)
        if self.last_prediction is not None:
            self.traffic_recieved_list.append(traffic_recieved)
            self.traffic_predicted_list.append(self.last_prediction)
        self.last_prediction = traffic_predicted

    def _record(self):
        if self.traffic_recieved_list or self.traffic_predicted_list:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"traffic_data_{timestamp}.csv"
            plot_filename = f"traffic_plot_{timestamp}.png"
            
            output_dir = os.path.join(root_dir, "data", "trafficPrediction")
            csv_path = os.path.join(output_dir, csv_filename)
            plot_path = os.path.join(output_dir, plot_filename)
            
            # Ensure the directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Write data to CSV
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Index', 'Traffic_Received', 'Traffic_Predicted'])
                
                # Write rows (handle case where lists might have different lengths)
                max_len = max(len(self.traffic_recieved_list), len(self.traffic_predicted_list))
                for i in range(max_len):
                    received = self.traffic_recieved_list[i] if i < len(self.traffic_recieved_list) else None
                    predicted = self.traffic_predicted_list[i] if i < len(self.traffic_predicted_list) else None
                    writer.writerow([i, received, predicted])
            
            print(f"Traffic data saved to: {csv_path}")
            print(f"Total records saved: {max_len}")
            
            # Create and save plot
            plt.figure(figsize=(12, 6))
            
            if self.traffic_recieved_list:
                plt.plot(self.traffic_recieved_list, label='Traffic Received', 
                        marker='o', linestyle='-', linewidth=2, markersize=4, alpha=0.7)
            
            if self.traffic_predicted_list:
                plt.plot(self.traffic_predicted_list, label='Traffic Predicted', 
                        marker='s', linestyle='--', linewidth=2, markersize=4, alpha=0.7)
            
            plt.xlabel('Time Step', fontsize=12)
            plt.ylabel('Traffic', fontsize=12)
            plt.title('Traffic Received vs Traffic Predicted', fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Traffic plot saved to: {plot_path}")
    
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
                    #==========================================================
                    #==================== Receive Data =========================
                    #==========================================================
                    data, addr = self.sock.recvfrom(self.buffer_size)
                    payload = data.decode()
                    src_ip, src_port = addr
                    #==========================================================
                    #==================== Online Predictor ====================
                    #==========================================================
                    self._onlinePredictor(payload)
                    #==========================================================
                    #==================== Print Info ==========================
                    #==========================================================
                    # Print received packet info and predicted traffic
                    print(f"--- Packet Received ---")
                    print(f"Source IP: {src_ip}:{src_port}")
                    print(f"Destination IP: {self.listen_ip}:{self.listen_port}")
                    print(f"Payload: {payload}")
                    print(f"--- Forwarding ---\n")
                    #==========================================================
                    #==================== Forwarding =========================
                    #==========================================================
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
        # Save traffic data to CSV before closing 
        self._record()
        if self.sock:
            self.sock.close()
            self.sock = None
        print("Relay stopped and traffic data saved")

if __name__ == "__main__":
    relay = RelayPredictor()
    relay.run()
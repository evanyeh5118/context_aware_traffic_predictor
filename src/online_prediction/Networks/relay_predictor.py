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

from src.online_prediction import ContextAwareOnlinePredictor, ContextFreeOnlinePredictor
from src.context_aware.models import createModel as createModel_context_aware
from src.context_free.models import createModel as createModel_context_free

configPath = "../../../experiments/config/combined_flows_forward_20.json"
modelFolder_context_aware = "../../../data/models/context_aware"
modelFolder_context_free = "../../../data/models/context_free"

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
        
        self.onlinePredictor_context_aware = None
        self.onlinePredictor_context_free = None
        self.last_prediction_context_aware = None
        self.last_prediction_context_free = None
        self.traffic_recieved_list = []
        self.traffic_predicted_list_context_aware = []
        self.traffic_predicted_list_context_free = []
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        self._initializePredictor()

    def _initializePredictor(self):
        config = json.load(open(configPath))
        name = config.get("NAME")
        len_window = config.get("LEN_WINDOW")

        with open(f"{modelFolder_context_aware}/{name}_modelConfig.pkl", "rb") as f:
            modelConfig_context_aware = pickle.load(f)
        with open(f"{modelFolder_context_aware}/{name}_metaConfig.pkl", "rb") as f:
            metaConfig_context_aware = pickle.load(f)
        with open(f"{modelFolder_context_free}/{name}_modelConfig.pkl", "rb") as f:
            modelConfig_context_free = pickle.load(f)
        with open(f"{modelFolder_context_free}/{name}_metaConfig.pkl", "rb") as f:
            metaConfig_context_free = pickle.load(f)
        metaConfig_context_aware.display()
        metaConfig_context_free.display()

        model_context_aware, _ = createModel_context_aware(modelConfig_context_aware)
        model_context_aware.load_checkpoint(f"{modelFolder_context_aware}/{name}.pth")
        model_context_free, _ = createModel_context_free(modelConfig_context_free)
        model_context_free.load_checkpoint(f"{modelFolder_context_free}/{name}.pth")
        
        self.onlinePredictor_context_aware = ContextAwareOnlinePredictor(model_context_aware, metaConfig_context_aware)
        self.onlinePredictor_context_free = ContextFreeOnlinePredictor(model_context_free, metaConfig_context_free)
    
    def signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully"""
        self.shutdown_flag = True
        print("\nShutdown signal received")

    def _updatePredictor(self, payload):
        payload = payload.split(",")
        payload = [float(x) for x in payload]
        self.onlinePredictor_context_aware.receive(payload)
        self.onlinePredictor_context_free.receive_signal()
        traffic_context_aware, traffic_recieved = self.onlinePredictor_context_aware.predict()
        traffic_context_free, _ = self.onlinePredictor_context_free.predict()
        traffic_context_aware = np.round(traffic_context_aware, 0).astype(int)
        traffic_context_free = np.round(traffic_context_free, 0).astype(int)
        if self.last_prediction_context_aware is not None and self.last_prediction_context_free is not None:
            self.traffic_recieved_list.append(traffic_recieved)
            self.traffic_predicted_list_context_aware.append(self.last_prediction_context_aware)
            self.traffic_predicted_list_context_free.append(self.last_prediction_context_free)
        self.last_prediction_context_aware = traffic_context_aware
        self.last_prediction_context_free = traffic_context_free

    def _record(self):
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
            writer.writerow(
                [
                    'Index', 
                    'Traffic_Received', 
                    'Traffic_Predicted_Context_Aware', 
                    'Traffic_Predicted_Context_Free',
                ]
            )
            # Write rows (handle case where lists might have different lengths)
            max_len = max(len(self.traffic_recieved_list), len(self.traffic_predicted_list_context_aware), len(self.traffic_predicted_list_context_free))
            for i in range(max_len):
                received = self.traffic_recieved_list[i] if i < len(self.traffic_recieved_list) else None
                predicted_context_aware = self.traffic_predicted_list_context_aware[i] if i < len(self.traffic_predicted_list_context_aware) else None
                predicted_context_free = self.traffic_predicted_list_context_free[i] if i < len(self.traffic_predicted_list_context_free) else None
                writer.writerow([i, received, predicted_context_aware, predicted_context_free])
        
        print(f"Traffic data saved to: {csv_path}")
        print(f"Total records saved: {max_len}")
        
        # Create and save plot
        plt.figure(figsize=(12, 6))
        
        if self.traffic_recieved_list:
            plt.plot(self.traffic_recieved_list, label='Traffic Received', 
                    marker='o', linestyle='-', linewidth=2, markersize=4, alpha=0.7)
        
        if self.traffic_predicted_list_context_aware:
            plt.plot(self.traffic_predicted_list_context_aware, label='Traffic Predicted Context Aware', 
                    marker='s', linestyle='--', linewidth=2, markersize=4, alpha=0.7)
        
        if self.traffic_predicted_list_context_free:
            plt.plot(self.traffic_predicted_list_context_free, label='Traffic Predicted Context Free', 
                    marker='s', linestyle='--', linewidth=2, markersize=4, alpha=0.7)
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Traffic', fontsize=12)
        plt.title('Traffic Received vs Traffic Predicted Context Aware vs Traffic Predicted Context Free', fontsize=14, fontweight='bold')
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
                    self._updatePredictor(payload)
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
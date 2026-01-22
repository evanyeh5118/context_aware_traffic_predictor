#!/usr/bin/env python3
import socket
import time
import argparse
import threading
import json
import sys
import os
import numpy as np
import signal
import csv
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to Python path for Docker/containerized execution
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.online_prediction import OnlinePredictor
from src.models import createModel
from src.config import MetaConfig, ModelConfig

# Get configuration from environment variables (set by launch script)
configPath = os.path.join(_project_root, "config", "motion_1ms_20.json")
modelFolder = os.path.join(_project_root, "model")
verbose_env = os.getenv('verbose', 0)

sent_packets = {}  # Store {seq: send_time} to calculate travel time
stop_event = threading.Event()
verbose = True  # Global verbose flag
startup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

class EdgePredictor:
    """Traffic predictor for edge node."""
    
    def __init__(self):
        """Initialize the traffic predictor with configuration."""
        self.onlinePredictor = None
        self.last_prediction = None
        self.traffic_received_list = []
        self.traffic_predicted_list = []
        self.last_trigger_time = None
        self.trigger_interval = None
        self.packet_count = 0
        
        self._initializePredictor()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals."""
        if verbose:
            print("\n[EDGE] Shutdown signal received")
        stop_event.set()
    
    def _initializePredictor(self):
        """Initialize the online predictor with model and config."""
        try:
            config = json.load(open(configPath))
            name = config.get("NAME")
            
            modelConfig = ModelConfig.load(f"{modelFolder}/{name}_modelConfig.json")
            metaConfig = MetaConfig.load(f"{modelFolder}/{name}_metaConfig.json")
            
            if verbose:
                metaConfig.display()
            
            model, _ = createModel(modelConfig)
            model.load_checkpoint(f"{modelFolder}/{name}.pth")
            
            self.onlinePredictor = OnlinePredictor(model, metaConfig)
            self.trigger_interval = config.get("SAMPLING_TIME") * config.get("LEN_WINDOW")
            self.last_trigger_time = time.time()
            
            if verbose:
                print("[PREDICTOR] Initialized successfully")
        except Exception as e:
            if verbose:
                print(f"[PREDICTOR ERROR] Failed to initialize: {e}")
    
    def update_predictor(self, payload):
        """Update predictor with received payload."""
        if self.onlinePredictor is None:
            return
        
        try:
            payload_values = payload.split(",")
            payload_values = [float(x) for x in payload_values]
            self.onlinePredictor.receive_signal()
            self.packet_count += 1
        except Exception as e:
            if verbose:
                print(f"[PREDICTOR ERROR] Failed to update: {e}")
    
    def trigger_prediction(self):
        """Trigger traffic prediction if interval has passed."""
        if self.onlinePredictor is None or self.last_trigger_time is None:
            return None
        
        if time.time() - self.last_trigger_time >= self.trigger_interval:
            self.last_trigger_time = time.time()
            try:
                traffic_predicted, traffic_received = self.onlinePredictor.predict()
                traffic_predicted = np.round(traffic_predicted, 0).astype(int)
                
                if self.last_prediction is not None:
                    self.traffic_received_list.append(traffic_received)
                    self.traffic_predicted_list.append(self.last_prediction)
                
                self.last_prediction = traffic_predicted
                
                if verbose:
                    print(f"[PREDICTION] Predicted: {traffic_predicted}, Received: {traffic_received}")
                
                return traffic_predicted
            except Exception as e:
                if verbose:
                    print(f"[PREDICTOR ERROR] Prediction failed: {e}")
                return None
        
        return None
    
    def save_results(self, timestamp):
        """Save traffic prediction results to CSV and plot."""
        csv_filename = f"traffic_data_{timestamp}.csv"
        plot_filename = f"traffic_plot_{timestamp}.png"
        
        output_dir = os.path.join(_project_root, "data", "capture")
        csv_path = os.path.join(output_dir, csv_filename)
        plot_path = os.path.join(output_dir, plot_filename)
        
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Write data to CSV
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Index', 'Traffic_Received', 'Traffic_Predicted'])
            
            # Write rows (handle case where lists might have different lengths)
            max_len = max(len(self.traffic_received_list), len(self.traffic_predicted_list))
            for i in range(max_len):
                received = self.traffic_received_list[i] if i < len(self.traffic_received_list) else None
                predicted = self.traffic_predicted_list[i] if i < len(self.traffic_predicted_list) else None
                writer.writerow([i, received, predicted])
        
        if verbose:
            print(f"[SAVE] Traffic data saved to: {csv_path}")
            print(f"[SAVE] Total records saved: {max_len}")
        
        # Create and save plot
        plt.figure(figsize=(12, 6))
        
        if self.traffic_received_list:
            plt.plot(self.traffic_received_list, label='Traffic Received', 
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
        
        if verbose:
            print(f"[SAVE] Traffic plot saved to: {plot_path}")
    
    def save_config(self, timestamp):
        """Save configuration to JSON file."""
        config_filename = f"config_{timestamp}.json"
        config_path = os.path.join(_project_root, "data", "capture", config_filename)
        
        try:
            config = json.load(open(configPath))
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            if verbose:
                print(f"[SAVE] Config saved to: {config_path}")
        except Exception as e:
            if verbose:
                print(f"[SAVE ERROR] Failed to save config: {e}")

# Initialize the traffic predictor
edge_predictor = EdgePredictor()

def receive_udp(listen_port, listen_ip=None):
    """Listen for incoming UDP packets, predict traffic pattern, and save results on shutdown."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    bind_addr = listen_ip if listen_ip else "0.0.0.0"
    sock.bind((bind_addr, listen_port))
    if verbose:
        print(f"[LISTEN] Started listening on {bind_addr}:{listen_port}")
        print(f"[LISTEN] Edge predictor active - will capture and predict traffic patterns")
    
    try:
        while not stop_event.is_set():
            try:
                sock.settimeout(1.0)
                data, addr = sock.recvfrom(1024)
                recv_time = time.time()
                message = data.decode('utf-8')
                parts = message.split(':', 1)
                if len(parts) == 2:
                    payload = parts[1]
                else:
                    print(f"[ERROR] Invalid message format from {addr}: {message} (expected series_num:payload)")
                    continue
                # =================================================================
                # ==================== Receive Data ==============================
                # =================================================================
                src_ip, src_port = addr
                edge_predictor.update_predictor(payload)
                
                # =================================================================
                # ==================== Online Predictor ==========================
                # =================================================================
                predicted = edge_predictor.trigger_prediction()
                
                # =================================================================
                # ==================== Print Info ================================
                # =================================================================
                # Try to extract sequence number from payload for additional info
                try:
                    seq = int(message)
                    if seq in sent_packets:
                        send_time = sent_packets[seq]
                        travel_time = (recv_time - send_time) * 1000  # Convert to milliseconds
                        if verbose:
                            print(f"[RECEIVED #{seq}] from {src_ip}:{src_port} | Travel time: {travel_time:.2f}ms")
                        del sent_packets[seq]  # Remove from tracking
                    else:
                        if verbose:
                            print(f"[RECEIVED #{seq}] from {src_ip}:{src_port}")
                except ValueError:
                    if verbose:
                        print(f"[RECEIVED] from {src_ip}:{src_port} | Payload: {payload}")
                
                if predicted is not None and verbose:
                    print(f"[PREDICTION TRIGGERED] Predicted traffic: {predicted}")
                    
            except socket.timeout:
                continue
            except Exception as e:
                if not stop_event.is_set():
                    if verbose:
                        print(f"[ERROR] Receive error: {e}")
                break
    
    except KeyboardInterrupt:
        if verbose:
            print("\n[LISTEN] Interrupted by user")
    finally:
        sock.close()
        if verbose:
            print(f"[LISTEN] Stopped listening on port {listen_port}")
        
        # Save results before closing
        if verbose:
            print("[EDGE] Saving captured traffic data and predictions...")
        edge_predictor.save_results(startup_timestamp)
        edge_predictor.save_config(startup_timestamp)
        
        if verbose:
            print(f"[EDGE] Total packets received: {edge_predictor.packet_count}")
            print(f"[EDGE] Total predictions made: {len(edge_predictor.traffic_predicted_list)}")
            print("[EDGE] Edge node shutting down")

def main():
    parser = argparse.ArgumentParser(
        description="Edge node: Listen for incoming UDP traffic and predict traffic patterns."
    )
    parser.add_argument("--listen-port", type=int, required=True, help="Port to listen on for incoming UDP packets")
    parser.add_argument("--listen-ip", help="IP address to listen on (default: 0.0.0.0)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()
    
    # Set global verbose flag
    global verbose
    verbose = args.verbose

    if verbose:
        print("[EDGE] Starting edge node with traffic prediction...")
        print(f"[EDGE] Listen port: {args.listen_port}")
        print(f"[EDGE] Listen IP: {args.listen_ip if args.listen_ip else '0.0.0.0'}")
        print(f"[EDGE] Session timestamp: {startup_timestamp}")
    
    # Start listening for UDP packets
    try:
        receive_udp(args.listen_port, args.listen_ip)
    except KeyboardInterrupt:
        if verbose:
            print("\n[EDGE] Keyboard interrupt received")
    except Exception as e:
        if verbose:
            print(f"[EDGE ERROR] Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()

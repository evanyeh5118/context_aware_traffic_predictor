## MVP Demo System (Minimum Function)

The MVP provides a **real-time traffic prediction demonstration** using a trained Seq2seq model with UDP-based network communication. This is a standalone deployment-ready system showcasing the practical application of the trained models.

**MVP vs Full System:**
- **Full System** (`src/`, `experiments/`): Complete training pipeline, data preprocessing, multiple model variants, and research notebooks
- **MVP** (`mvp/`): Production-ready inference engine with network simulation for demonstration and deployment

### MVP Features

- **Online Inference**: Real-time traffic prediction on streaming data
- **Network Simulation**: UDP sender → relay → receiver architecture
- **Pre-trained Model**: Ready-to-use model for forward motion prediction
- **Adaptive Gain**: Online gain optimization for prediction accuracy
- **Visualization**: Real-time traffic monitoring and prediction plots

### MVP Quick Start

#### Option 1: Jupyter Notebook Demo (Recommended)

Run the interactive demo notebook:

```bash
cd mvp
jupyter notebook online_inference.ipynb
```

The notebook demonstrates:
- Loading pre-trained Seq2seq model
- Processing streaming traffic data
- Real-time prediction with adaptive gain
- Visualization of predicted vs actual traffic

#### Option 2: Network Simulation

Run the full UDP network simulation with three components:

**Terminal 1 - Receiver** (receives final data):
```bash
cd mvp/src/network
python receiver.py
```

**Terminal 2 - Relay** (processes & forwards data):
```bash
cd mvp/src/network
export RELAY_IP=127.0.0.1
export RELAY_PORT=5000
python relay.py
```

**Terminal 3 - Sender** (sends traffic data):
```bash
cd mvp/src/network
export CSV_FILE=../../data/sender/traffic_dataset.csv
export RELAY_IP=127.0.0.1
export RELAY_PORT=5000
python sender.py
```

**Using Launch Script** (Linux/Mac):
```bash
cd mvp/scripts
chmod +x launch_all.sh
./launch_all.sh
```

### MVP Configuration

#### Model Configuration (`mvp/config/config.json`)

```json
{
  "NAME": "combined_flows_forward",
  "LEN_WINDOW": 30,
  "TRAIN_RATIO": 0.7,
  "SAMPLING_TIME": 0.01,
  "CONTEXT_IDXS": [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15],
  "DIM_DATA": 12,
  "DPDR_PARAMS": {"dbParameter": 0.01, "alpha": 0.01, "mode": "fixed"}
}
```

#### Network Configuration (`mvp/scripts/config.yaml`)

Key parameters:
- `relay.ip` / `relay.port`: Relay server address
- `replay.real_timing`: Enable timing-based replay (0=fast, 1=real-time)
- `replay.time_scale`: Speed multiplier (e.g., 2.0 = 2x faster)
- `settings.use_predictor`: Enable ML predictions (0=off, 1=on)
- `settings.duration_sec`: Simulation duration (0=until data ends)

### MVP Architecture

```
┌─────────────┐  UDP    ┌──────────────┐  UDP    ┌───────────────┐
│   Sender    │────────>│    Relay     │────────>│   Receiver    │
│ (CSV Data)  │  5000   │ (Predictor)  │  5001   │ (Monitor)     │
└─────────────┘         └──────────────┘         └───────────────┘
                              │
                              ├─ Load Seq2seq Model
                              ├─ Process Traffic Data
                              ├─ Predict Future Traffic
                              └─ Log & Visualize
```

### MVP Models

Pre-trained model included:
- **Model**: `mvp/model/combined_flows_forward.pth`
- **Architecture**: Seq2seq (LSTM encoder-decoder)
- **Input**: 30-window historical traffic (12-dim context)
- **Output**: Next window traffic prediction
- **Configurations**: 
  - `combined_flows_forward_metaConfig.json`
  - `combined_flows_forward_modelConfig.json`

### Results
- Model outputs: `Results/models/`
  - `context_assisted/`: Context-assisted model predictions
  - `context_free/`: Context-free model predictions
- Evaluation metrics and visualizations are generated in the notebooks

### Tips

**General:**
- **GPU**: Training automatically uses CUDA if available for faster processing
- **Performance**: Start with smaller `lenWindow` values for quicker validation
- **Paths**: Always run notebooks from the project root directory

**MVP Specific:**
- **Quick Demo**: Start with `online_inference.ipynb` for a quick demonstration
- **Network Testing**: Test individual components (sender/receiver) before running full simulation
- **Timing**: Use `time_scale` in config.yaml to speed up or slow down replay
- **Windows Users**: Use PowerShell or WSL for running the launch script, or run components individually
- **Port Conflicts**: If ports 5000/5001 are in use, modify them in config files and scripts

### Citation
If you use this code or results in academic work, please cite appropriately (add your citation here).

### License

MIT License - See LICENSE file for details.

### Acknowledgements
This repository implements traffic prediction models for wireless sensor networks with support for context-aware prediction.

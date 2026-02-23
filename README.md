## Context-Aware Traffic Predictor

Wireless sensor networks (WSNs) transmit motion data over constrained links, so predicting future traffic volumes lets relay nodes allocate resources proactively. This project explores two complementary strategies: a **context-free** (CF) approach that learns traffic patterns purely from historical transmission counts via an encoder–decoder Seq2Seq LSTM, and a **context-aware** (CA) approach that also ingests the motion sensor readings (the "context") to anticipate how deadband data-reduction will shape future traffic. A deadband-reduction preprocessor, online UDP-based inference pipeline, and Markov traffic-pattern analyser round out the toolkit.

### Project Structure

```
context_aware_traffic_predictor/
├── src/
│   ├── context_free/           # Seq2Seq LSTM encoder-decoder model
│   ├── context_aware/          # Context-aware predictor (D2T + ContextAdjuster)
│   ├── dataset_manager/        # Raw data reading, deadband reduction, DataUnit
│   ├── online_prediction/      # Real-time UDP sender → relay → receiver pipeline
│   ├── pcapfile_generation/    # PCAP capture and modification utilities
│   ├── base/                   # Base model with checkpoint save/load
│   ├── analysis/               # Analysis helper tools
│   └── Markov/                 # Markov traffic-pattern modelling
├── experiments/
│   ├── notebooks/              # Jupyter notebooks (see below)
│   └── config/                 # JSON experiment configurations
├── data/
│   ├── raw/                    # Raw motion data (Task0/1/2)
│   ├── processed/dpdr/         # Deadband-reduced CSV datasets
│   └── models/                 # Trained model checkpoints & configs
├── mvp/                        # Minimal viable-product demo
├── setup.py
├── requirements.txt
└── README.md
```

### Environment

- Python 3.9 (tested)
- Recommended: Conda environment

```bash
conda create -n traffic_predictor_3_9 python=3.9 -y
conda activate traffic_predictor_3_9
pip install -r requirements.txt
python -m ipykernel install --user --name traffic_predictor_3_9   # optional
```

### Data

The project expects raw motion files at `data/raw/Task{0,1,2}/exp*/motion.txt`. Keep the default layout and notebooks will resolve paths automatically. Processed datasets are written to `data/processed/dpdr/`.

---

### Notebooks

All notebooks live under `experiments/notebooks/` and are organised by workflow stage.

#### 1. Dataset Creation — `dataset/` & `create_dataset/`

| Notebook | Purpose |
|----------|---------|
| `dataset/main_create_traffic_dataset.ipynb` | Batch-converts raw motion files into deadband-reduced CSV datasets using experiment configs from `experiments/config/`. Outputs go to `data/processed/dpdr/`. |
| `create_dataset/gen_dataset_context_free.ipynb` | Generates datasets formatted specifically for context-free model training (sliding-window traffic sequences). |

#### 2. Training — `training/`

| Notebook | Purpose |
|----------|---------|
| `training/verify_tarffic_pattern.ipynb` | Loads raw data, applies deadband reduction, and visualises forward/backward traffic patterns to sanity-check data quality and compression rates before training. |
| `training/train_context_free.ipynb` | End-to-end context-free pipeline: loads config → creates DataUnit → preprocesses sliding windows → trains Seq2Seq LSTM → evaluates and plots predictions. Saves model (`.pth`) and configs (`.json`) to `data/models/context_free/`. |
| `training/train_context_aware.ipynb` | End-to-end context-aware pipeline: loads config → creates DataUnit → preprocesses with normalisation and optional filtering → saves processed CSV and configs (`.pkl`) to `data/models/context_aware/`. |
| `training/smooth_context_aware.ipynb` | Compares smoothing strategies (Kalman filter vs. derivative-based smoothing) on context-aware input features. Useful for tuning the preprocessing filter before a full training run. |

#### 3. Inference — `inference/`

| Notebook | Purpose |
|----------|---------|
| `inference/offline_inferencec_context_free.ipynb` | Loads a trained CF model and runs batch inference on a test set. Plots actual vs. predicted traffic and exports predictions to CSV. |
| `inference/online_inference_context_free.ipynb` | Simulates real-time inference: reads CSV traffic row-by-row, feeds each signal to `ContextFreeOnlinePredictor`, and triggers predictions at window intervals. Visualises received vs. predicted traffic over time. |
| `inference/online_inference_context_aware.ipynb` | Same real-time simulation for the CA model: receives data points with payload, applies exponential filtering and gain optimisation via `ContextAwareOnlinePredictor`, and plots results. |
| `inference/verify_setup.ipynb` | Verifies the CA `DataProcessor` setup — loads CSV and meta config, adds data points, and checks window feature shapes (context, last-transmission source, non-smoothed context). |

#### 4. Analysis — `analysis/`

| Notebook | Purpose |
|----------|---------|
| `analysis/main_udp_files.ipynb` | Analyses captured UDP packet files: parses traffic traces, computes statistics, and visualises traffic patterns. |

---

### Results

- Trained models and configs: `data/models/context_free/` and `data/models/context_aware/`
- Evaluation metrics and plots are generated inline within the notebooks

### Tips

- **GPU** — Training uses CUDA automatically when available.
- **Quick iteration** — Start with a smaller `LEN_WINDOW` (e.g., 20) in the experiment config.
- **Paths** — Run notebooks from the project root so relative imports resolve correctly.

### Citation

If you use this code or results in academic work, please cite appropriately.

### License

MIT License — See LICENSE file for details.
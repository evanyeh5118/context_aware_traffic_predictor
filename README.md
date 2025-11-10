## Context-Aware Traffic Predictor

A neural network-based traffic prediction system for wireless sensor networks, implementing both **context-free** (CF) and **context-assisted** (CA) prediction models. The project includes data preprocessing, model training, evaluation, and Markov modeling for traffic pattern analysis.

### Project Structure

```
context_aware_traffic_predictor/
├── src/
│   ├── traffic_predictor/      # Main prediction models (CF & CA)
│   ├── context_free/           # Context-free model implementation
│   ├── DatasetManager/         # Dataset processing and deadband reduction
│   └── Markov/                 # Markov modeling utilities
├── data/
│   ├── raw/                    # Raw motion data (Task0/1/2)
│   └── processed/              # Processed datasets
├── experiments/
│   └── notebooks/              # Jupyter notebooks for experiments
├── Results/
│   └── models/                 # Trained model outputs
└── requirements.txt            # Python dependencies
```

### Environment
- Python 3.9 (tested)
- Recommended: Conda environment

Create and activate environment:
```bash
conda create -n traffic_predictor_3_9 python=3.9 -y
conda activate traffic_predictor_3_9
```

Install required packages:
```bash
pip install -r requirements.txt
```

Create Jupyter kernel (optional):
```bash
python -m ipykernel install --user --name traffic_predictor_3_9
```

### Data
The project expects data under `data/raw/`:
- `data/raw/Task{0,1,2}/exp*/motion.txt`
- Optionally `data/processed/processed_data_multiTask.txt` for aggregated processing

You do not need to move files if you keep the default structure. The notebooks and dataset converters expect this layout.

### Usage

#### Step 1: Verify Traffic Patterns
Run `experiments/notebooks/main00_verify_tarffic_pattern.ipynb`

This notebook:
- Loads raw motion data from the dataset
- Applies deadband reduction preprocessing
- Visualizes traffic patterns for forward and backward motions
- Verifies data quality and compression rates

**Key parameters:**
- `dbParameter`: Deadband parameter (e.g., 0.01, 0.05)
- `alpha`: Smoothing parameter
- `direction`: "forward" or "backward"

#### Step 2: Train and Evaluate Models
Run `experiments/notebooks/main01_train_traffic_predictor.ipynb`

This notebook:
- Prepares training/test datasets with configurable window sizes
- Trains context-assisted (CA) traffic prediction models
- Evaluates model performance on test data
- Visualizes prediction results

**Key parameters:**
- `lenWindow`: Sequence window length (default: 20)
- `trainRatio`: Train/test split ratio
- `dataAugment`: Enable data augmentation
- `direction`: Motion direction to process

**Outputs:**
- Trained model parameters saved to `Results/models/`
- Evaluation results (actual vs predicted traffic)

The training automatically uses GPU if available (CUDA), otherwise falls back to CPU.

### Results
- Model outputs: `Results/models/`
  - `context_assisted/`: Context-assisted model predictions
  - `context_free/`: Context-free model predictions
- Evaluation metrics and visualizations are generated in the notebooks

### Tips
- **GPU**: Training automatically uses CUDA if available for faster processing
- **Performance**: Start with smaller `lenWindow` values for quicker validation
- **Paths**: Always run notebooks from the project root directory

### Citation
If you use this code or results in academic work, please cite appropriately (add your citation here).

### License

MIT License - See LICENSE file for details.

### Acknowledgements
This repository implements traffic prediction models for wireless sensor networks with support for context-aware prediction.
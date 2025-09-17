## Context-Aware Traffic Predictor

This repository contains code and notebooks for building and evaluating a context-aware traffic predictor with both context-free (CF) and context-assisted (CA) models. It includes dataset preprocessing, model training/evaluation, and figure generation for analysis and paper-ready plots.

### Project structure
- `libs/`
  - `TrafficGenerator/`: dataset conversion and preprocessing utilities
  - `TrafficPredictor/`
    - `ContextFree/`: CF data prep, model, training, evaluation
    - `ContextAssisted/`: CA data prep, enhanced model, training, evaluation
    - `HelperFunctions.py`: helper utilities
  - `MarkovModel/`: utilities for thresholding, grouping, and Markov modeling
  - `HelperFunc.py`: shared helpers (e.g., filename encoding)
- `Dataset/`: raw and processed dataset folders (Task0/1/2, per-exp motion.txt)
- `Results/TrafficPredictor/`: model params and evaluation outputs
- Notebooks:
  - `main00_verify_tarffic_pattern.ipynb`
  - `main01_train_traffic_predictor.ipynb`
  - `figure01_traffic_prediction.ipynb`
  - `figure02_markov_modeling.ipynb`
- `Archived/`: older or exploratory notebooks

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
pip install numpy matplotlib torch
```
Optional (for development):
```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name traffic_predictor_3_9
```

### Data
The project expects data under `Dataset/`:
- `Dataset/Task{0,1,2}/exp*/motion.txt`
- Optionally `Dataset/processed_data_multiTask.txt` for aggregated processing

You do not need to move files if you keep the default structure. The notebooks and `libs.TrafficGenerator.DatasetConvertor` expect this layout.

### How to run

#### 1) Verify traffic patterns
Open and run:
- `main00_verify_tarffic_pattern.ipynb`

This inspects raw motion/traffic patterns and basic preprocessing.

#### 2) Train and evaluate models
Open and run:
- `main01_train_traffic_predictor.ipynb`

Key notes:
- Select DB parameters, direction, and mode at the top (e.g., `dbParams`, `direction`, `mode`, `alpha`, `lenWindow_list`, `train_ratio`).
- The notebook trains both CF and CA variants by calling:
  - CF: `libs.TrafficPredictor.ContextFree` (PreparingDataset, createModel, trainModelByDefaultSetting, evaluateModel)
  - CA: `libs.TrafficPredictor.ContextAssisted` (PreparingDataset, createModel, trainModelByDefaultSetting, evaluateModel)
- Training uses GPU if available (prints `Used device: cuda`); otherwise CPU.

Outputs saved to:
- `Results/TrafficPredictor/evaluate/{CF|CA}/..._train.pkl`
- `Results/TrafficPredictor/evaluate/{CF|CA}/..._test.pkl`
- CA model params (for reproducibility): `Results/TrafficPredictor/modelParams/...txt.pkl`

Filestem naming follows helpers:
- `libs.HelperFunc.encode_float_filename`, `decode_float_filename`

#### 3) Generate analysis figures
- `figure01_traffic_prediction.ipynb`: loads saved `.pkl` results, computes metrics (MSE, weighted F1 via `libs.compute_weighted_f1_score` and `libs.MarkovModel` thresholds), and plots comparison bar graphs. It can export high-res figures (update the path at the end of the notebook for your environment).
- `figure02_markov_modeling.ipynb`: Markov modeling and analysis (balanced thresholds and group assignments via `libs.MarkovModel`).

### Results and artifacts
- Evaluation: `Results/TrafficPredictor/evaluate/{CF|CA}/...`
- Model params: `Results/TrafficPredictor/modelParams/`
- Example exports in notebooks save to a user path (update paths as needed).

### Tips and troubleshooting
- GPU: PyTorch will automatically use CUDA if available. If running on CPU, expect slower training; reduce `lenWindow_list` or set fewer epochs in the training functions if needed.
- Long runs: Both CF/CA training can be compute-intensive. Start with `lenWindow=10`, `mode="fixed"`, small train splits to validate setup.
- Paths: Notebooks use relative paths under the project root. Ensure you start Jupyter in the repository root.

### Citation
If you use this code or results in academic work, please cite appropriately (add your citation here).

### License
Add your license information here (e.g., MIT, Apache-2.0).

### Acknowledgements
This repository builds on internal modules under `libs/` for dataset processing, model training, and Markov modeling.
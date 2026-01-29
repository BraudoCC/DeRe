# DeReF: Deconstruction and Reconstruction Framework for Adaptive Multivariate Time Series Forecasting (Official Implementation)

This repository provides the official implementation of **DeReF (Deconstruction and Reconstruction Framework)**, an adaptive framework designed for automated Multivariate Time Series Forecasting (MTSF). 

DeReF shifts the paradigm from static model selection to dynamic architecture synthesis by:
1. **Deconstruction**: Decomposing monolithic SOTA models into a unified, fine-grained component pool.
2. **Reconstruction**: Adaptively assembling specialized forecasting models based on predictability-aware deep meta-features.

---

## 📂 Code Structure

The project is organized as follows:

- `data_provider/`: Data loading and preprocessing utilities for various benchmarks.
- `exp/`: Core experimental logic, including meta-training and evaluation loops.
- `layers/`: Implementation of modular components (e.g., normalization, attention, decomposition).
- `meta/`: Predictability-aware meta-feature extraction based on TabPFN.
- `models/`: Architecture definitions for reconstructed models and backbones.
- `utils/`: Common utilities for logging, metrics, and visualization.
- `run_meta_dl.py`: Main entry point for meta-learner training and adaptive selection.
- `run.py`: Script for training and evaluating individual model configurations.

---

<!-- ## 🛠️ Environment Setup

To set up the environment, we recommend using Conda: -->

<!-- ```bash
# Create and activate environment
conda create -n deref python=3.9
conda activate deref

# Install dependencies
pip install -r requirements.txt -->
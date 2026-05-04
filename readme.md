# DIAGNO: Diagonal Spherical Neural Operators for Heterogeneous Earth Dynamics Modeling

This repository provides a complete research framework for global climate and oceanic forecasting within the field of AI for Science (AI4S). It includes the official implementation of the **DIAGNO** (Diagonal Spherical Neural Operator) model, training datasets, distributed training code, and comprehensive parameter configurations.

## 🌟 Overview

Unlike standard neural operators that assume rotation equivariance, **DIAGNO** is designed to capture the heterogeneous dynamics of the Earth system. The model explicitly seeks to break rotation equivariance assumptions to shift the paradigm toward explicit cross-modal interaction. It isolates zonal and meridional interactions by fusing spectral components along the diagonal where $l-m = \text{constant}$.

## 🚀 Quick Start

To initiate the training pipeline, ensure your environment meets the requirements (Python, PyTorch, xarray, Cartopy) and execute the provided bash script. This script automatically orchestrates the Distributed Data Parallel (DDP) environment:
```bash
bash run_train.sh
```

## ⚙️ Configuration Guide

The project separates the training hardware/flow configuration from the specific model architecture configuration.

### 1. Training Parameters (`run_train.sh`)
Global execution settings and hyperparameters are managed within the shell script:
* **`ROOT_PATH`**: The output directory where model checkpoints, logs, and statistics will be saved.
* **`MODELS`**: A space-separated list of model variants to train sequentially (e.g., `diagno_e128`).
* **`EPOCHS`**: Defines both `PRETRAIN_EPOCHS` (1-step) and `FINETUNE_EPOCHS` (2-step autoregressive).
* **`LEARNING RATES`**: Independent learning rates for pretraining and finetuning.
* **`Hardware Setup`**: Update `CUDA_VISIBLE_DEVICES` and `nproc_per_node` to match your multi-GPU or Slurm setup.

### 2. Model Architecture (`model_registry`)
The specific architectural parameters (e.g., embedding dimensions, hidden layers) for DiagNO variants are defined in the `model_registry` module. Adjust these definitions before execution to modify the model capacity.

## 📊 Data Availability

This repository balances accessibility with the large-scale nature of Earth system reanalysis:
* **SSWE Dataset**: The synthetic dataset for **Spherical Shallow Water Equations** experiments is provided within this package for immediate reproducibility.
* **ERA5 & GLORYS12**: Due to their extreme size, global reanalysis datasets such as **ERA5** (atmospheric) and **GLORYS12** (oceanic) are not hosted here.
* **Download Instructions**: Users should download these datasets from their respective official portals, such as the Copernicus Climate Data Store or the Copernicus Marine Service.

## 📁 Output Structure

The training pipeline automatically organizes results within the `ROOT_PATH`:
* `best_model_pretrain.pt` / `best_model_finetune.pt`: Optimized weights for each stage.
* `latest_model.pt`: Checkpoint for seamless recovery from hardware interruptions.
* `history.csv`: Per-epoch logs for Loss, MAE, MSE, and RMSE metrics.

## 📚 Code References
Our implementation integrates and adapts specialized components from the following open-source projects:
* **Neural-Solver-Library**: https://github.com/thuml/Neural-Solver-Library
* **Torch-Harmonics**: https://github.com/NVIDIA/torch-harmonics

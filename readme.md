# DIAGNO: Diagonal Spherical Neural Operators for Heterogeneous Earth Dynamics Modeling

This repository provides a complete research framework for global climate and oceanic forecasting within the field of AI for Science (AI4S). It includes the official implementation of the **DIAGNO** (Diagonal Spherical Neural Operator) model, training datasets, distributed training code, and comprehensive parameter configurations.

## Overview

Unlike standard neural operators that assume rotation equivariance, **DIAGNO** is designed to capture the heterogeneous dynamics of the Earth system. The model explicitly seeks to break rotation equivariance assumptions to shift the paradigm toward explicit cross-modal interaction. It isolates zonal and meridional interactions by fusing spectral components along the diagonal where $l-m = \text{constant}$.

## Quick Start

To initiate the training pipeline, ensure your environment meets the requirements (Python, PyTorch, xarray, Cartopy) and execute the provided bash script. This script automatically orchestrates the Distributed Data Parallel (DDP) environment:

```bash
bash run_train.sh
```

## Configuration Guide

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

## Data Availability

This repository balances accessibility with the large-scale nature of Earth system reanalysis:
* **SSWE Dataset**: The synthetic dataset for **Spherical Shallow Water Equations** experiments is provided within this package for immediate reproducibility.
* **ERA5 & GLORYS12**: Due to their extreme size, global reanalysis datasets such as **ERA5** (atmospheric) and **GLORYS12** (oceanic) are not hosted here.
* **Download Instructions**: Users should download these datasets from their respective official portals, such as the Copernicus Climate Data Store or the Copernicus Marine Service.

## Output Structure

The training pipeline automatically organizes results within the `ROOT_PATH`:
* `best_model_pretrain.pt` / `best_model_finetune.pt`: Optimized weights for each stage.
* `latest_model.pt`: Checkpoint for seamless recovery from hardware interruptions.
* `history.csv`: Per-epoch logs for Loss, MAE, MSE, and RMSE metrics.

## Code References
Our implementation integrates and adapts specialized components from the following open-source projects:
* **Neural-Solver-Library**: [https://github.com/thuml/Neural-Solver-Library](https://github.com/thuml/Neural-Solver-Library)
* **Torch-Harmonics**: [https://github.com/NVIDIA/torch-harmonics](https://github.com/NVIDIA/torch-harmonics)

---

## DIAGNO Environment Setup Guide

This document provides step-by-step instructions for configuring the Python environment required to run the DIAGNO model. The environment is built on Python 3.10 and includes PyTorch, PyTorch Geometric (PyG), geospatial libraries, and specific data access APIs.

### 🛠️ Prerequisites

Before starting, please ensure your system meets the following requirements:
* **Package Manager**: Anaconda or Miniconda is installed.
* **Hardware & Drivers**: An NVIDIA GPU is available.
* **Important**: Ensure your installed NVIDIA display driver supports CUDA 12.1.

### Installation Steps

**Step 1: Create and Activate the Conda Environment**  
First, create a new conda environment named `diagno_env` with Python 3.10, and activate it:

```bash
conda create -n diagno_env python=3.10 -y
conda activate diagno_env
```

**Step 2: Install Core Scientific and Geospatial Packages**  
Use Conda to install fundamental libraries for scientific computing, parallel processing, and geospatial data handling:

```bash
conda install -y \
    numpy scipy pandas matplotlib scikit-learn \
    h5py netcdf4 xarray dask \
    mpi4py cmake \
    cartopy shapely pyproj \
    pytz tqdm
```

**Step 3: Install PyTorch (CUDA 12.1)**  
Install PyTorch 2.4.0 and its vision/audio extensions. We specifically pull from the official PyTorch wheel index compiled for CUDA 12.1:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

**Step 4: Install PyTorch Geometric (PyG) Dependencies**  
Install the underlying graph neural network operations.

```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

**Step 5: Install Auxiliary Tools, Storage, and Data APIs**  
Finally, use pip to install the remaining machine learning utilities, cloud storage libraries, 3D visualization tools, and APIs for fetching meteorological/oceanic datasets:

```bash
# Core ML utilities, web tools, and experiment tracking
pip install \
    aiohttp anyio attrs boto3 botocore timm \
    einops filelock fsspec huggingface-hub \
    jinja2 joblib networkx pillow protobuf \
    pydantic requests rich sympy tqdm typer wandb

# 3D visualization and chunked data storage
pip install vtk
pip install zarr numcodecs

# Astronomy and climate/ocean data APIs
pip install astropy
pip install copernicusmarine cdsapi
```

### Environment Configuration File

All required libraries and dependencies for this environment are also listed in the `environment.yml` file included in this repository. You can check this file for a comprehensive reference of the environment configuration.

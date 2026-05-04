# Spherical Shallow Water Equations (SWE) Training

This repository contains the training code and configuration scripts for modeling Spherical Shallow Water Equations (SWE) using PyTorch. 

## 🚀 Quick Start

To start the training process, simply execute the provided bash script. This script automatically handles the Distributed Data Parallel (DDP) environment setup and starts the training loop:
```bash
bash run_train.sh
```

## ⚙️ Configuration Guide

To make experiments manageable and reproducible, this project separates the training hardware/flow configuration from the specific model architecture configuration.

### 1. Training Parameters (`run_train.sh`)
The global training flow, hardware settings, and hyperparameters are controlled directly within the `run_train.sh` script. Open this file to modify:
* **`ROOT_PATH`**: The output directory where model checkpoints, logs, and stats will be saved.
* **`MODELS`**: A space-separated list of models to train sequentially (e.g., `"diagno_e128 diagno_e64"`).
* **`EPOCHS`**: Define `PRETRAIN_EPOCHS` (1-step training) and `FINETUNE_EPOCHS` (2-step autoregressive training).
* **`LEARNING RATES`**: Independent learning rates for pretraining (`PRETRAIN_LR`) and finetuning (`FINETUNE_LR`).
* **`BATCH_SIZE`**: Training batch size per GPU.
* **`Hardware Setup`**: Update `CUDA_VISIBLE_DEVICES` and `nproc_per_node` to match your local multi-GPU setup.

### 2. Model Architecture (`model_registry`)
The specific architectural parameters for each neural network model (e.g., hidden layers, network dimensions, specific block configurations) are defined in the `model_registry` module. 

If you need to adjust a model's internal structure or add a new model variant, modify the specific definitions within the `model_registry` codebase before executing the training script.

## 📁 Output Structure
During training, the script will automatically create the `ROOT_PATH` directory and generate subfolders for each model. Inside each model's directory, you will find:
* `best_model_pretrain.pt` / `best_model_finetune.pt`: The model weights with the lowest validation loss.
* `latest_model.pt`: Checkpoint for resuming training.
* `pretrain_history.csv` / `finetune_history.csv`: Training metrics (Loss, MAE, MSE, RMSE) logged per epoch.

## 📚 Code References
Our implementation utilizes and adapts code from the following open-source libraries:
* **[Neural-Solver-Library]**: https://github.com/thuml/Neural-Solver-Library
* **[Torch-Harmonics]**: https://github.com/NVIDIA/torch-harmonics
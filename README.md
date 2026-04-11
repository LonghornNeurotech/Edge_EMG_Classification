# EMG Gesture Recognition on Edge
**Developed by the Edge AI Team @ Longhorn Neurotech**

## Overview

This repository contains a machine learning pipeline for classifying hand gestures using 8-channel forearm EMG signals. The project focuses on taking a model from raw data exploration to optimized hardware deployment.

We trained a **Multilayer Perceptron (MLP)** on time-domain engineered features and applied **Static 8-bit Quantization (INT8)** to the final model. The quantized model is exported to **ONNX format** for efficient, low-latency inference on edge devices like the Raspberry Pi Zero 2 W.

---

## Repository Contents

- **Notebooks:** Full pipeline including EDA, preprocessing, feature extraction, model training, Optuna hyperparameter tuning, and ONNX export.
- **Deployment:** `run_inference.py` script to benchmark and run the quantized ONNX model locally on a Raspberry Pi.
- **Models:** Exported FP32 and INT8 `.onnx` models for edge evaluation.

> **Note:** The raw UCI EMG dataset is excluded from this repository due to size constraints.

---

## Edge Deployment Guide (Raspberry Pi Zero 2 W)

The following steps were used to deploy and test the model locally on the Raspberry Pi.

### 1. Download Model Artifacts

First, install `gdown` to download the model files and test data from Google Drive:

```bash
pip install gdown --break-system-packages
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# Download the deployment folder
gdown --folder "https://drive.google.com/drive/folders/YOUR_DRIVE_LINK_HERE"
` `` 

### 2. Setup Python Virtual Environment

To avoid `externally-managed-environment` (PEP 668) restrictions on the latest Raspberry Pi OS, create and activate a virtual environment:

` ``bash
python3 -m venv emg_env
source emg_env/bin/activate
` ``

### 3. Install Dependencies

Install the required ONNX Runtime and NumPy packages inside the virtual environment:

` ``bash
pip install onnxruntime numpy
` ``

### 4. Run Inference

Navigate to the downloaded folder, ensure `run_inference.py` is present, and execute the benchmark:

` ``bash
cd Pi_Deployment
python run_inference.py
` ``

---

## Benchmark Results (Raspberry Pi Zero 2 W)

Running inference on the test set (**4,497 samples**) yielded the following performance metrics:

| Model Version | Accuracy | Total Time | Avg Latency/Sample |
|---|---|---|---|
| FP32 Baseline (`emg_mlp_model.onnx`) | 90.50% | 4.68 seconds | 1.04 ms |
| INT8 Quantized (`emg_mlp_model_quantized.onnx`) | 90.37% | 1.94 seconds | 0.43 ms |

---

## Conclusion

Static 8-bit quantization achieved a **2.4x speedup** (cutting latency from 1.04ms to 0.43ms) with a negligible accuracy drop of only **0.13%**.

# AlignAD

> AlignAD: Self-Supervised Robust Multimodal Alignment for End-to-End Autonomous Driving

## Table of Contents

- [1. Environment Setup](#1-environment-setup)
- [2. Dataset Preparation](#2-dataset-preparation)
- [3. Training](#3-training)
- [4. Evaluation](#4-evaluation)
- [5. Results](#5-results)

---

## 1. Environment Setup

```bash
conda create -n alignad python=3.8
conda activate alignad
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e ./nuplan-devkit
pip install -e .
```

## 2. Dataset Preparation

### 2.1 Download the dataset

Follow the official [NAVSIM dataset instructions](https://github.com/autonomousvision/navsim/blob/main/docs/install.md) to download "maps", "navsim_logs", and "sensor_blobs".

### 2.2 Build the dataset cache for training

```bash
bash ./scripts/training/run_train_metric_cache.sh
bash ./scripts/training/run_dataset_cache.sh
```

### 2.3 Build the metric cache for evaluation

```bash
bash ./scripts/evaluation/run_metric_caching.sh
```

## 3. Training

```bash
bash ./scripts/training/run_alignad.sh 
```

During training, you can visualize the loss and scores by launching TensorBoard with:

```bash
tensorboard --logdir $NAVSIM_EXP_ROOT/training_alignad_agent/timestamp/lightning_logs/version_0
```

## 4. Evaluation

You need to set the weight path in the run_alignad.sh file before starting the evaluation.

```bash
bash ./scripts/evaluation/run_alignad.sh
```

## 5. Results


| Method  | Backbone  | NC   | DAC  | TTC  | Comfort | EP   | PDMS | PDMS (Best-of-N) | Weight Download                                           |
| ------- | --------- | ---- | ---- | ---- | ------- | ---- | ---- | ---------------- | --------------------------------------------------------- |
| AlignAD | ResNet-34 | 98.1 | 98.3 | 94.3 | 99.8    | 89.1 | 91.8 | 98.7             | [Hugging Face](https://huggingface.co/Anderewuuu/AlignAD) |



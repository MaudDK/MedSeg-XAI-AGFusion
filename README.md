# MedSeg-XAI-AGFusion

Medical image segmentation experiments with CNN, ViT, Swin, and dual-encoder attention fusion models, focused on Kvasir-SEG (and partially CAMUS) workflows.

## Overview

This repository contains:
- Training pipelines for multiple segmentation backbones.
- Attention-guided dual-encoder fusion variants (ResNet + Swin).
- Benchmarking utilities for segmentation metrics and inference timing.
- Attention map export for model explainability.

## Implemented model families

- **CNN baselines:** Res34 U-Net, DeepLabV3-ResNet50 U-Net, DuckNet.
- **Transformer baselines:** ViT (tiny/small/base/large/huge), Swin variants, timm ViT variants.
- **Fusion models:** `DualEncoder`, `AttDualEncoder`, `WeightedAttDualEncoder`, `AttentionDualEncoderSwin`, `AttentionDualEncoderRes`.

## Repository structure

- `MAIN_MODEL_TRAINER.py` – main training entry point (Kvasir).
- `CAMUS_MODEL_TRAINER.py` – CAMUS-oriented training script.
- `Benchmark.py` – model benchmarking + optional attention visualization export.
- `trainer.py` – generic trainer with checkpointing and history serialization.
- `*.py` – model definitions and loss/metric utilities.
- `commands.sh` – example SLURM training commands.

## Environment setup

Create a Python environment and install core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision timm pandas numpy tqdm matplotlib opencv-python
```

> Dataset loaders are imported from `datasets.KVASIRData` and `datasets.CAMUSData`, so those modules/files must be available in your environment.

## Data layout (expected)

The scripts expect CSV split files and images under a `datasets` directory, e.g.:

```text
datasets/
  Kvasir-SEG/
    Kvasir_dataset.csv
    images/
    masks/
  CAMUS/
    CAMUS_dataset.csv
```

## Training

Run training from the repository root:

```bash
python MAIN_MODEL_TRAINER.py --model AttentionDualEncoderSwin --epochs 500 --patience 300
```

### Available `--model` options

- `Res34Unet`
- `Res34UnetNoSkip`
- `DeepLabV3Res50UNetNoSkip`
- `DeepLabV3Res50UNet`
- `ViT_Tiny`
- `ViT_Small`
- `ViT_Base`
- `Deit_Base`
- `Swin_Base`
- `Swin_Base_Skip`
- `ViT_Base_Tim`
- `ViT_Small_Tim`
- `ViT_Tiny_Tim`
- `ViT_Large`
- `ViT_Huge`
- `DuckNet`
- `DualEncoder`
- `AttDualEncoder`
- `WeightedAttDualEncoder`
- `AttentionDualEncoderSwin`
- `AttentionDualEncoderRes`

For cluster execution examples, see:
- `commands.sh`

## Benchmarking and XAI outputs

Run:

```bash
python Benchmark.py
```

This will:
- evaluate configured checkpoints,
- compute metrics (Dice, IoU, Precision, Recall, Accuracy, F1, BCE),
- measure inference time,
- write results to `benchmark_results.csv`,
- and export attention visualizations to `attention_weights/` (for attention-enabled models).

## Outputs

- **Checkpoints:** saved under `checkpoints/<model>/<date>/<session_id>/...`
- **Training history:** pickled history files in the same checkpoint tree.
- **Benchmark results:** `benchmark_results.csv`
- **Attention maps:** `attention_weights/`

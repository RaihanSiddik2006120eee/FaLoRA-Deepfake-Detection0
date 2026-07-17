# FaLoRA: Frequency-Aware Low-Rank Adaptation for Deepfake Detection

FaLoRA is a parameter-efficient deepfake detection framework that combines spatial features from a pretrained Vision Transformer with frequency-domain forensic information.

The model uses a frozen CLIP-pretrained Vision Transformer backbone and introduces frequency-conditioned Low-Rank Adaptation modules into the query and value projections of the transformer attention blocks.

Instead of fully fine-tuning the backbone, FaLoRA learns a small number of task-specific parameters. A frequency-aware gate controls the contribution of individual LoRA rank directions according to the frequency characteristics of each input image.

> **Research status:** This repository contains the cleaned core implementation of an ongoing research project. Dataset preparation scripts, training scripts, checkpoints, and complete reproduction instructions will be added after the manuscript process is completed.

---

## Method Overview

FaLoRA contains four main components:

1. **Frozen Vision Transformer backbone**
   - Uses a CLIP-pretrained ViT-B/16 model.
   - Original backbone parameters remain frozen.

2. **Frequency feature extractor**
   - Extracts DCT amplitude information.
   - Extracts FFT phase information.
   - Extracts SRM noise residuals.
   - Combines nine frequency-domain channels into a compact frequency token.

3. **Frequency-gated LoRA**
   - Applies LoRA updates to the query and value projections.
   - Combines the frequency token with the detached CLS token.
   - Generates an input-dependent gate for each LoRA rank direction.
   - Allows the strength of adaptation to vary across input images.

4. **Classification and projection heads**
   - Predicts real or manipulated images.
   - Produces normalized embeddings for supervised contrastive learning.

---

## Core Idea

Standard LoRA applies the same low-rank adaptation mechanism to every input. FaLoRA makes the adaptation input-dependent.

For each image, the frequency extractor produces a compact forensic representation. This representation is combined with the spatial CLS token to generate rank-level gates. These gates control how strongly the LoRA updates modify the frozen transformer representation.

This design aims to preserve the pretrained backbone while allowing stronger adaptation when manipulation-related frequency artifacts are present.

---

## Repository Structure

```text
FaLoRA-Deepfake-Detection/
├── src/
│   ├── __init__.py
│   ├── evaluation.py
│   ├── falora_attention.py
│   ├── frequency_extractor.py
│   ├── model.py
│   └── training_utils.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

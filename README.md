# CDDRec Implementation

PyTorch implementation of **CDDRec (Conditional Denoising Diffusion for Sequential Recommendation)** from the paper "Conditional Denoising Diffusion for Sequential Recommendation" (arXiv:2304.11433v1).

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Run Smoke Tests

Verify the implementation works:

```bash
python tests/test_basic.py
```

### 2. Prepare Data

The data should be in JSON format:

```json
{
  "train": {
    "sequences": [[1, 2, 3], [4, 5, 6]],
    "targets": [4, 7]
  },
  "val": {
    "sequences": [[1, 2], [4, 5]],
    "targets": [3, 6]
  },
  "test": {
    "sequences": [[1, 2, 3], [4, 5, 6, 7]],
    "targets": [4, 8]
  },
  "num_users": 100,
  "num_items": 1000
}
```

Use `cddrec/data/preprocessing.py` utilities to process raw interaction data.

### 3. Train Model

```bash
python scripts/train.py \
  --data_path data/processed/dataset.json \
  --num_items 1000 \
  --embedding_dim 128 \
  --diffusion_steps 30 \
  --batch_size 128 \
  --lr 0.001 \
  --num_epochs 100
```

### 4. Evaluate Model

```bash
python scripts/evaluate.py \
  --data_path data/processed/dataset.json \
  --num_items 1000 \
  --checkpoint checkpoints/best_model.pth \
  --split test
```

## Sanity Check (Recommended First Step)

Before training on real data, verify the implementation with pattern-based synthetic data:

```bash
# Automated sanity check (~10-15 minutes on CPU)
bash scripts/run_sanity_check.sh
```

This creates synthetic data with verifiable patterns (sequential tracks and item clusters), trains a small model, and verifies the model learned the patterns.

**Success criteria:**
- Sequential Rank@5: >70% (model predicts next item in sequence)
- Cluster Top-5: >60% (model predicts items from same cluster)

See `SANITY_CHECK_GUIDE.md` for detailed explanation, troubleshooting, and interpreting results.

## Project Structure

```
seqdiff/
├── cddrec/                      # Main package
│   ├── models/                  # Model components
│   │   ├── encoder.py           # Sequence encoder
│   │   ├── decoder.py           # Denoising decoder
│   │   ├── diffuser.py          # Diffusion process
│   │   └── cddrec.py            # Main model
│   ├── losses.py                # Loss functions
│   ├── training.py              # Training loop
│   ├── utils.py                 # Metrics and utilities
│   └── data/                    # Data handling
│       ├── dataset.py           # PyTorch Dataset
│       ├── augmentation.py      # Data augmentation
│       └── preprocessing.py     # Preprocessing utilities
├── scripts/
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation script
└── tests/
    └── test_basic.py            # Smoke tests
```

## Model Architecture

The model consists of three main components:

1. **Sequence Encoder**: Self-attention based encoder for historical interactions
2. **Conditional Denoising Decoder**: Cross-attention decoder that generates predictions
3. **Step-Wise Diffuser**: Manages forward and reverse diffusion process

## Key Features

- Multi-step denoising for high-quality predictions
- Cross-divergence loss for ranking-aware generation
- Multi-view contrastive learning for robustness
- Conditional generation based on historical interactions

## Training Configuration

Hyperparameters (from paper):
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 128
- Embedding dimension: 128
- Max sequence length: 20
- Dropout: 0.2
- Diffusion steps: 10-30 (dataset-dependent)

See `CLAUDE.md` for detailed implementation notes.

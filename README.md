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

Process your raw interaction data using the preprocessing API:

```python
from cddrec.data.preprocessing import preprocess_interactions

# Your raw data: (user_id, item_id, timestamp) tuples
# user_id/item_id can be any type (str, int, etc.)
interactions = [
    ("user_abc", "item_xyz", 1000.0),
    ("user_abc", "item_123", 1001.0),
    # ...
]

# Preprocess and save (handles filtering, splitting, ID mapping)
result = preprocess_interactions(
    interactions,
    min_user_interactions=5,  # Minimum interactions per user
    min_item_interactions=5,  # Minimum interactions per item
    output_path="data/processed/my_dataset.json"
)

# Also works with generators for memory efficiency:
def stream_from_db():
    for row in db.execute("SELECT user_id, item_id, timestamp FROM interactions"):
        yield (row[0], row[1], row[2])

result = preprocess_interactions(stream_from_db(), output_path="data/processed/my_dataset.json")
```

The API automatically:
- Converts IDs to model format (0-indexed users, 1-indexed items)
- Filters users/items by interaction count
- Splits into train/val/test sets
- Saves data and ID mappings for later use

### 3. Train Model

```bash
# Use the preprocessed data (num_items is read from the data file)
python scripts/train.py \
  --data_path data/processed/my_dataset.json \
  --embedding_dim 128 \
  --diffusion_steps 30 \
  --batch_size 128 \
  --lr 0.001 \
  --num_epochs 100
```

### 4. Evaluate Model

```bash
python scripts/evaluate.py \
  --data_path data/processed/my_dataset.json \
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

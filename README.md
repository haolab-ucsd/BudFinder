# BudFinder

A deep learning pipeline for detecting cell division (budding) events in *Saccharomyces cerevisiae* time-lapse microscopy.

## Overview

Identifying cell division events in time-lapse microscopy of budding yeast is labor-intensive when performed manually and becomes impractical at high throughput. BudFinder automates this task using a two-stage deep learning approach:

1. **Self-supervised pre-training**: A Masked Autoencoder (MAE) learns spatial representations of individual cell crops by reconstructing images from which 75% of patches have been randomly masked. This stage requires no division labels.

2. **Supervised classification**: The pre-trained MAE encoder is transferred to a temporal classifier (CELLDIV_MAE). For each frame in an 11-frame temporal stack (t-5 through t+5), the encoder produces a CLS token embedding. These 11 embeddings are augmented with sinusoidal positional encodings and passed through a frame-level transformer that outputs a binary division/non-division prediction.

Three pre-trained division prediction models are included.

## Repository Structure

```
BudFinder/
├── BudFinder/
│   ├── Models.py                 # MAE_2d and CELLDIV_MAE architectures
│   ├── Datasets.py               # PyTorch Dataset classes for training
│   ├── Utils.py                  # Image cropping, stack creation, EMA, schedulers
│   ├── train_mae.py              # Stage 1: MAE pre-training script
│   ├── train.py                  # Stage 2: division classifier training script
│   ├── predict.py                # Prediction via remote Flask server
│   ├── predict_offline.py        # Local prediction (binary output)
│   ├── predict_offline_prob.py   # Local prediction (with probabilities)
│   ├── training_config.yaml      # Division classifier training configuration
│   ├── training_mae_config.yaml  # MAE training configuration
│   ├── requirements.txt          # Python dependencies
│   ├── Movies.txt                # Movie identifiers used during training
│   ├── MAEWeights/               # Pre-trained MAE encoder components
│   │   ├── MAE_encoder_epoch10.pth
│   │   ├── MAE_init_enc_epoch10.pth
│   │   └── MAE_pos_emb_epoch10.pth
│   └── OfflineModels/            # Pre-trained division prediction models
│       ├── mae_div_complete.pt
│       ├── mae_div_complete2.pt
│       └── mae_div_complete3.pt
└── example/
    ├── csv/
    │   └── divstack_cleaned.csv  # Example cell tracking and division annotations
    └── movies/
        ├── xy01c1.tif            # Example TIFF microscopy time-lapse
        ├── xy02c1.tif
        └── xy03c1.tif
```

## Installation

Requires Python 3.9+.

```bash
git clone <repository-url>
cd BudFinder
python3 -m venv venv
source ./venv/bin/activate
pip install -r BudFinder/requirements.txt
```

CUDA GPU recommended.

## Data Format

### Input: TIFF Movies

Single-channel grayscale TIFF image stacks (one file per time-lapse movie). Files must follow the naming convention `xy<NN>c1.tif`, where `<NN>` is the movie number (e.g., `xy01c1.tif`, `xy12c1.tif`). The pipeline filters for filenames containing `c1.tif`.

### Input: CSV

A CSV file containing cell tracking data derived from Cellpose segmentation. Required columns:

| Column | Type | Description |
|---|---|---|
| `Movie` | int | Movie number matching the `<NN>` in the TIFF filename |
| `Frame` | int | Frame number (1-indexed) |
| `tracks` | int | Cell track identifier (consistent across frames for the same cell) |
| `Centroid_X` | float | X coordinate of cell centroid in pixels |
| `Centroid_Y` | float | Y coordinate of cell centroid in pixels |
| `div` | int | Division label: `1` = division, `0` = non-division *(training only)* |
| `idx_glob` | int | Global cell index for train/validation splitting *(training only)* |

For prediction, only `Movie`, `Frame`, `tracks`, `Centroid_X`, and `Centroid_Y` are required.

### Output: CSV

**`predict_offline.py`** produces `div_predictions.csv`:

| Column | Description |
|---|---|
| `movie` | Movie number |
| `track_man` | Cell track identifier |
| `num_divs` | Number of detected division events |
| `div_frames` | Frame numbers where divisions were detected |

**`predict_offline_prob.py`** produces `div_prediction_probabilities.csv` with the same schema, except `div_frames` is replaced by `div_probabilities` (per-frame division probability scores).

## Quick Start: Running on the Example Dataset

This walks through running the pre-trained model on the included example data.

**1. Edit the prediction script paths:**

Open `BudFinder/predict_offline.py` and find the `main()` function at the bottom of the file. Replace the three path variables with paths to the included example data:

```python
def main():
    movie_folder_path = '../example/movies'
    csv_path = '../example/csv/divstack_cleaned.csv'
    offline_model_path = './OfflineModels/mae_div_complete.pt'
    full_pipeline(movie_folder_path, csv_path, offline_model_path)
```

- `movie_folder_path` -- folder containing the `.tif` movie files
- `csv_path` -- the Cellpose-derived tracking CSV
- `offline_model_path` -- one of the three pre-trained models in `OfflineModels/`

**2. Run prediction:**

```bash
cd BudFinder/BudFinder
python predict_offline.py
```

The script will process each movie in the folder: load the TIFF stack, crop each tracked cell at its centroid, build 11-frame temporal stacks, and run the model on every stack.

**3. Output:**

When finished, a file named `div_predictions.csv` will be written to the current directory (`BudFinder/BudFinder/`). Each row corresponds to one tracked cell and contains:
- `movie` -- the movie number
- `track_man` -- the cell track identifier
- `num_divs` -- how many division events were detected
- `div_frames` -- the frame numbers at which divisions were detected

To also get per-frame division probability scores, use `predict_offline_prob.py` instead (same setup, edit `main()` the same way). Its output is `div_prediction_probabilities.csv`, which includes a `div_probabilities` column with the model's confidence at every frame.

Three pre-trained TorchScript models are provided in `OfflineModels/` (`mae_div_complete.pt`, `mae_div_complete2.pt`, `mae_div_complete3.pt`).

> **Note:** `predict.py` communicates with a remote Flask server for inference and is not intended for standalone use.

## Running on Your Own Data

To run on your own data, you need:
1. Single-channel TIFF time-lapse movies following the `xy<NN>c1.tif` naming convention (see [Data Format](#data-format) above).
2. A Cellpose-derived tracking CSV with at least the columns `Movie`, `Frame`, `tracks`, `Centroid_X`, and `Centroid_Y`.

Edit the paths in `predict_offline.py` (or `predict_offline_prob.py`) to point to your data and run as above.

## Training from Scratch

Training is a two-stage process. Both scripts accept a `--config_dir` argument pointing to a YAML configuration file.

### Stage 1: MAE Pre-training

The MAE learns general image representations from unlabeled cell crops via masked reconstruction. No division labels are needed for this stage.

**1. Edit `training_mae_config.yaml`:**

Set the two paths under `data:` to point to your data:
```yaml
data:
 tifpath: '/path/to/your/movies'    # folder containing xy<NN>c1.tif files
 csvpath: '/path/to/your/data.csv'  # Cellpose tracking CSV
```

**2. Run:**

```bash
cd BudFinder/BudFinder
python train_mae.py --config_dir training_mae_config.yaml
```

**3. Output:**

Weights are saved to `MAEWeights_test/` every `save_freq` epochs (default: every 2). Each checkpoint produces three files:
- `MAE_encoder_epoch<N>.pth` -- transformer encoder
- `MAE_init_enc_epoch<N>.pth` -- initial patch encoding MLP
- `MAE_pos_emb_epoch<N>.pth` -- learned positional embeddings

These three files are required as input for Stage 2.

### Stage 2: Division Classifier

The classifier loads the pre-trained MAE encoder from Stage 1 and trains a frame-level transformer to predict division events from 11-frame temporal context.

**1. Edit `training_config.yaml`:**

Set the data paths and point to the MAE weights from Stage 1:
```yaml
data:
 moviepath: '/path/to/your/movies'
 csvpath: '/path/to/your/data.csv'
mae_model:
 mae_init_enc_path: 'MAEWeights_test/MAE_init_enc_epoch10.pth'
 mae_pos_emb_path: 'MAEWeights_test/MAE_pos_emb_epoch10.pth'
 mae_enc_path: 'MAEWeights_test/MAE_encoder_epoch10.pth'
```

The CSV must include the `div` column (division labels) and `idx_glob` column (for train/validation splitting) for this stage.

**2. Run:**

```bash
cd BudFinder/BudFinder
python train.py --config_dir training_config.yaml
```

**3. Output:**

Model weights and a TorchScript model (`mae_div_torchscript.pt`) are saved to `MAEDIVWeights/` every `save_freq` epochs (default: every 2). The TorchScript model can be used directly with the prediction scripts. Training metrics are logged to TensorBoard.
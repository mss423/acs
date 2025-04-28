# Comparing Downsampling Methods for BERT Classification

## Description

This project investigates the impact of different data downsampling techniques (Random, Stratified, ACS) on the performance of BERT-based classifiers for natural language tasks. We apply various sampling methods to reduce the size of a dataset before training a `simpletransformers` classification model and evaluate the resulting model performance.

## Table of Contents

- [Comparing Downsampling Methods for BERT Classification](#comparing-downsampling-methods-for-bert-classification)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [Directory Structure](#directory-structure)
  - [Installation](#installation)
  - [Data](#data)
  - [Configuration](#configuration)
  - [Usage](#usage)
    - [Downsampling Methods](#downsampling-methods)
    - [Training](#training)
    - [Prediction](#prediction)
  - [Contributing](#contributing)
  - [License](#license)

## Directory Structure

```
<repository_name>/
├── .gitignore           # Files/folders ignored by Git
├── LICENSE              # Project license (e.g., MIT)
├── README.md            # This file
├── requirements.txt     # Python dependencies
│
├── data/                # Data files (potentially gitignored)
│   ├── raw/             # Original data (place input CSVs here)
│   ├── processed/       # Intermediate/cleaned data (optional)
│   └── downsampled/     # Generated data subsets (optional output)
│
├── notebooks/           # Jupyter notebooks for exploration/analysis (optional)
│
├── src/                 # Source code
│   ├── __init__.py      # Makes src a Python package
│   ├── config.py        # Configuration settings (paths, hyperparameters, etc.)
│   ├── data_loader.py   # Loads and prepares datasets using a Dataset class
│   ├── downsampling.py  # Implements downsampling methods (random, stratified, acs)
│   ├── train.py         # Main script for training models
│   └── predict.py       # Script for making predictions with trained models
│
└── results/             # Output files (gitignored by default)
    ├── models/          # Saved model checkpoints (output of train.py)
    ├── predictions/     # Prediction outputs (output of predict.py)
    ├── logs/            # Training logs (optional)
    └── cache/           # Cache directory for transformers/simpletransformers
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data

* Place your raw training and testing data (e.g., `train.csv`, `test.csv`) inside the `data/raw/` directory.
* The data is expected to be in CSV format (or specify the separator in `config.py`).
* Ensure your data files contain the text and label columns specified in `src/config.py` under `DATASET_CONFIG['text_col']` and `DATASET_CONFIG['label_col']`.
* The `data_loader.py` script handles loading and basic preprocessing, renaming the specified columns to `text` and `labels` internally for compatibility with `simpletransformers`.

## Configuration

All major settings are controlled via `src/config.py`:

* **Paths:** Data directories, results directories, cache directory.
* **Dataset:** `DATASET_NAME` (e.g., 'custom_csv'), `DATASET_CONFIG` (file paths, column names, separator). **Modify these to match your data.**
* **Downsampling:** `DOWNSAMPLING_METHOD` ('random', 'stratified', 'acs', 'none'), `TARGET_SAMPLE_SIZE`. Additional parameters for specific methods (like ACS) can be added here and passed via `**extra_sampling_args` in `downsampling.py`.
* **Model:** `MODEL_TYPE` ('bert', 'roberta', etc.), `MODEL_NAME` (e.g., 'bert-base-uncased').
* **`MODEL_ARGS`:** A dictionary containing hyperparameters and settings for `simpletransformers` (epochs, learning rate, batch size, sequence length, output directories, evaluation settings, etc.). Adjust these based on your needs and hardware.
* **General:** `RANDOM_SEED` for reproducibility.

## Usage

### Downsampling Methods

The following methods are available in `src/downsampling.py` and can be selected in `src/config.py` via `DOWNSAMPLING_METHOD`:

* `none`: Use the full training dataset.
* `random`: Simple random sampling to `TARGET_SAMPLE_SIZE`.
* `stratified`: Stratified sampling based on the label column to `TARGET_SAMPLE_SIZE`. Requires `scikit-learn`.
* `acs`: Active Class Selection (or similar user-defined method). Requires implementation within `downsampling.py`. Can accept additional parameters (e.g., `coverage`, `sim_thresh`) via `config.py` or command line if `train.py`/`downsampling.py` are modified to accept them.

### Training

The `src/train.py` script handles data loading, downsampling (based on config), model training, and evaluation.

1.  **Configure:** Ensure `src/config.py` is set up correctly for your dataset, desired downsampling method, and model hyperparameters.
2.  **Run Training:**
    ```bash
    python src/train.py
    ```
    The script will:
    * Load data specified in `config.py`.
    * Apply the configured downsampling method to the training data.
    * Initialize the `simpletransformers` model based on `config.py`.
    * Train the model, saving checkpoints and potentially evaluating on a validation/test set during training (controlled by `MODEL_ARGS`).
    * Perform a final evaluation on the test set (if provided) after training.
    * Outputs (models, results) will be saved in the directories specified in `config.py` (usually within `results/`).

### Prediction

The `src/predict.py` script uses a trained model checkpoint to make predictions on new, unlabeled data.

1.  **Prepare Input:** Create a CSV file containing the text data you want predictions for. Ensure it has the text column specified by the `--text_col` argument.
2.  **Run Prediction:**
    ```bash
    python src/predict.py \
        --input_file path/to/your/unlabeled_data.csv \
        --model_dir path/to/your/trained_model_checkpoint \
        --output_file path/to/save/predictions.csv \
        [--text_col your_text_column_name] \
        [--batch_size 64]
    ```
    * `--input_file`: (Required) Path to the CSV with data to predict.
    * `--model_dir`: (Optional) Path to the saved model checkpoint directory (e.g., `results/models/best_model` or `results/models/checkpoint-1000-epoch-1`). Defaults to the `output_dir` specified in `config.py`.
    * `--output_file`: (Optional) Path to save the predictions CSV. Defaults to a file in `results/predictions/`.
    * `--text_col`: (Optional) Name of the text column in the input file. Defaults to 'text'.
    * `--batch_size`: (Optional) Batch size for prediction. Defaults to 32.

    The output CSV will contain the original data along with a new `predicted_label` column.


# src/config.py

import os
from pathlib import Path

# --- Project Root ---
# Determine the project root directory dynamically
# Assumes config.py is in src/ folder, so root is one level up
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
print(f"Project Root determined as: {PROJECT_ROOT}")

# --- Path Configuration ---
# Define paths relative to the project root for portability
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOWNSAMPLED_DATA_DIR = DATA_DIR / "downsampled"

RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_OUTPUT_DIR = RESULTS_DIR / "models"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
LOGS_DIR = RESULTS_DIR / "logs"
CACHE_DIR = RESULTS_DIR / "cache" # For simpletransformers/huggingface cache

# --- Ensure directories exist ---
# Create directories if they don't exist to avoid errors later
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
DOWNSAMPLED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --- Model Configuration (for simpletransformers) ---
MODEL_TYPE = "bert"             # Model type (e.g., 'bert', 'roberta', 'distilbert')
MODEL_NAME = "bert-base-uncased" # Specific pre-trained model checkpoint

# Classification Arguments for simpletransformers
# See simpletransformers docs for all available options:
# https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model
MODEL_ARGS = {
    "output_dir": str(MODEL_OUTPUT_DIR), # Convert Path object to string
    "cache_dir": str(CACHE_DIR),         # Convert Path object to string

    # --- Training Control ---
    # "num_train_epochs": 3,
    "learning_rate": 4e-5,
    # "train_batch_size": 16, # Adjust based on GPU memory
    "gradient_accumulation_steps": 1, # Increase if batch size is too large for memory
    # "max_seq_length": 128, # Max token length; adjust based on your data and model limits

    # --- Evaluation ---
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 500, # Evaluate every N steps
    "evaluate_during_training_verbose": True,
    "eval_batch_size": 32,

    # --- Saving and Logging ---
    "save_steps": -1, # Save model only at the end of epochs (-1). Set > 0 to save every N steps.
    "save_eval_checkpoints": True, # Save best model based on evaluation
    "save_model_every_epoch": True,
    "logging_steps": 50, # Log metrics every N steps
    "use_early_stopping": True,
    "early_stopping_delta": 0.01,
    "early_stopping_metric": "eval_loss", # Metric to monitor for early stopping
    "early_stopping_metric_minimize": True, # True if lower metric is better (loss)
    "early_stopping_patience": 3, # Stop if no improvement after N evaluations

    # --- Hardware and Performance ---
    "use_multiprocessing": True, # For data loading
    "use_cuda": True, # Set to False if you don't have a GPU or want to use CPU
    "fp16": False, # Set to True for mixed-precision training (requires compatible GPU/CUDA setup)

    # --- Reproducibility ---
    # "manual_seed": 42,
    "overwrite_output_dir": True, # Set to False to avoid overwriting previous results

    # Add any other simpletransformers args you need
}

# --- General Settings ---
RANDOM_SEED = 42 # Global random seed for numpy, random, etc. where needed outside of model args

# --- You can add more sections as needed ---
# E.g., Evaluation metrics, specific preprocessing flags, etc.

print("Configuration loaded.")

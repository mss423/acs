# src/predict.py

import pandas as pd
import os
import argparse # For command-line arguments
import torch

# Import project modules
import config # Your configuration file

# Import simpletransformers
from simpletransformers.classification import ClassificationModel

def predict(model_dir: str, input_file: str, output_file: str, text_col: str, batch_size: int):
    """
    Loads a trained model and makes predictions on new data.

    Args:
        model_dir (str): Path to the directory containing the trained
                         simpletransformers model checkpoint (e.g., results/models/...).
        input_file (str): Path to the input CSV file containing the text data
                          to predict on.
        output_file (str): Path to save the output CSV file with predictions.
        text_col (str): Name of the column in the input CSV containing the text.
        batch_size (int): Batch size for prediction.
    """
    print("--- Starting Prediction Script ---")
    print(f"Model Directory: {model_dir}")
    print(f"Input File: {input_file}")
    print(f"Output File: {output_file}")
    print(f"Text Column: {text_col}")
    print(f"Batch Size: {batch_size}")

    # --- 1. Validate Inputs ---
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        return
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # --- 2. Load Trained Model ---
    print("\n--- Loading Model ---")
    try:
        # Check CUDA availability based on config (or default to True if not set)
        use_cuda = config.MODEL_ARGS.get("use_cuda", True) and torch.cuda.is_available()
        print(f"Using CUDA: {use_cuda}")

        # Load the model from the specified directory
        # We need the model_type from the config used during training
        model = ClassificationModel(
            model_type=config.MODEL_TYPE,
            model_name=model_dir, # Pass the directory path here
            use_cuda=use_cuda,
            args={'eval_batch_size': batch_size} # Use eval_batch_size for prediction
        )
        print(f"Loaded model type '{config.MODEL_TYPE}' from {model_dir}")

    except Exception as e:
        print(f"Error loading model from {model_dir}: {e}")
        import traceback
        print(traceback.format_exc())
        return

    # --- 3. Load Input Data ---
    print("\n--- Loading Input Data ---")
    try:
        input_df = pd.read_csv(input_file)
        if text_col not in input_df.columns:
            print(f"Error: Text column '{text_col}' not found in input file {input_file}.")
            return
        # Handle potential missing text values if necessary
        input_df.dropna(subset=[text_col], inplace=True)
        input_df[text_col] = input_df[text_col].astype(str) # Ensure text is string

        texts_to_predict = input_df[text_col].tolist()
        print(f"Loaded {len(texts_to_predict)} texts to predict from {input_file}.")
        if not texts_to_predict:
             print("Error: No valid texts found in the input file after processing.")
             return

    except Exception as e:
        print(f"Error reading or processing input file {input_file}: {e}")
        return

    # --- 4. Make Predictions ---
    print("\n--- Making Predictions ---")
    try:
        # model.predict returns predictions and raw model outputs (probabilities/logits)
        predictions, raw_outputs = model.predict(texts_to_predict)
        print(f"Generated {len(predictions)} predictions.")

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        print(traceback.format_exc())
        return

    # --- 5. Format and Save Output ---
    print("\n--- Saving Predictions ---")
    try:
        # Create an output DataFrame
        output_df = input_df.copy() # Keep original data
        output_df['predicted_label'] = predictions

        # Optionally add raw outputs (e.g., probabilities) if needed
        # The structure of raw_outputs depends on the model type
        # For classification, it's often numpy array of logits or probabilities
        # output_df['raw_outputs'] = list(raw_outputs) # Example

        output_df.to_csv(output_file, index=False)
        print(f"Predictions saved successfully to {output_file}")

    except Exception as e:
        print(f"Error formatting or saving output file {output_file}: {e}")

    print("\n--- Prediction Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using a trained simpletransformers model.")

    parser.add_argument(
        "--model_dir",
        type=str,
        required=False, # Make optional, default to config
        default=config.MODEL_ARGS.get("output_dir", "results/models"), # Default from config
        help="Path to the directory containing the trained model checkpoint."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input CSV file containing text data."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=False, # Make optional, default based on input
        default=None,
        help="Path to save the output CSV file with predictions."
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="text", # Default column name
        help="Name of the column in the input CSV containing the text."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32, # Default prediction batch size
        help="Batch size to use for prediction."
    )

    args = parser.parse_args()

    # Determine default output file path if not provided
    if args.output_file is None:
        input_basename = os.path.basename(args.input_file)
        input_name, input_ext = os.path.splitext(input_basename)
        # Default output path in the PREDICTIONS_DIR from config
        args.output_file = os.path.join(config.PREDICTIONS_DIR, f"{input_name}_predictions.csv")


    # Resolve model directory relative to project root if it's not absolute
    if not os.path.isabs(args.model_dir):
        model_dir_resolved = os.path.join(config.PROJECT_ROOT, args.model_dir)
    else:
        model_dir_resolved = args.model_dir

    predict(
        model_dir=model_dir_resolved,
        input_file=args.input_file,
        output_file=args.output_file,
        text_col=args.text_col,
        batch_size=args.batch_size
    )

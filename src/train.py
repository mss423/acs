# src/train.py

import pandas as pd
import numpy as np
import random
import torch # Often needed for setting seeds or checking CUDA
import os
import json # For saving results
from typing import Optional

# Import project modules
import config # Your configuration file
from data_loader import Dataset
from downsampling import apply_downsampling

# Import simpletransformers
from simpletransformers.classification import ClassificationModel
from simpletransformers.ner import NERArgs, NERModel
from sklearn.model_selection import train_test_split # For creating validation set if needed
from sklearn.metrics import classification_report, accuracy_score, f1_score # For evaluation

CROSSNER_LABELS = ['B-researcher', 'I-researcher', 'O', 'B-product', 'B-algorithm',
       'I-algorithm', 'B-conference', 'I-conference', 'B-field',
       'I-field', 'B-metrics', 'B-location', 'I-location', 'B-country',
       'I-metrics', 'I-country', 'B-person', 'I-person', 'B-programlang',
       'B-organisation', 'B-university', 'I-university', 'B-misc',
       'I-misc', 'B-task', 'I-task', 'I-product', 'I-organisation',
       'I-programlang']


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Set random seed to {seed}")


def train(
    train_df: pd.DataFrame,
    num_labels: int,
    eval_df: Optional[pd.DataFrame] = None,
    **kwargs
):
    """
    Initializes, trains, and evaluates the simpletransformers model.

    Args:
        train_df (pd.DataFrame): The DataFrame to be used for training
                                 (potentially already downsampled).
                                 Must contain 'text' and 'labels' columns.
        num_labels (int): The number of unique labels in the dataset.
        eval_df (Optional[pd.DataFrame], optional): The DataFrame for evaluation
                                                    during/after training.
                                                    Defaults to None.
    """
    print("\n--- Entering Training Function ---")
    print(f"Training data size: {len(train_df)}")
    if eval_df is not None:
        print(f"Evaluation data size: {len(eval_df)}")
    else:
        print("No evaluation data provided to train function.")

    # --- 1. Initialize Model ---
    print("\n--- Initializing Model ---")
    # Check CUDA availability against config
    use_cuda = config.MODEL_ARGS.get("use_cuda", True) and torch.cuda.is_available()
    if config.MODEL_ARGS.get("use_cuda", True) and not torch.cuda.is_available():
        print("Warning: CUDA specified in config but not available. Using CPU.")
    elif not config.MODEL_ARGS.get("use_cuda", True):
         print("CUDA usage disabled in config. Using CPU.")
    else:
        print("CUDA available and enabled. Using GPU.")

    try:
        args = config.MODEL_ARGS
        args['num_train_epochs'] = kwargs.get('num_train_epochs', 3)
        args['train_batch_size'] = kwargs.get('train_batch_size', 16)
        args['manual_seed']      = kwargs.get('manual_seed', 42)
        is_crossner              = kwargs.get('is_crossner', False)

        if is_crossner:
            args['labels_list'] = CROSSNER_LABELS
            model = NERModel(
                model_type=config.MODEL_TYPE,
                model_name="bert-base-cased",
                args=args,
                use_cuda=use_cuda

            )
        else:
            model = ClassificationModel(
                model_type=config.MODEL_TYPE,
                model_name=config.MODEL_NAME,
                num_labels=num_labels,
                args=args,#config.MODEL_ARGS,
                use_cuda=use_cuda # Use the checked value
            )
        print(f"Model Type: {config.MODEL_TYPE}")
        print(f"Model Name: {config.MODEL_NAME}")
        print(f"Number of Labels: {num_labels}")
        print(f"Output Directory: {config.MODEL_ARGS.get('output_dir')}")

    except Exception as e:
        print(f"Error initializing ClassificationModel: {e}")
        return

    # --- 2. Train Model ---
    print("\n--- Starting Model Training ---")
    try:
        # The train_model method handles evaluation during training if configured
        # and if eval_df is provided
        model.train_model(
            train_df=train_df,
            eval_df=eval_df # Pass the prepared validation/test set
        )
        print("--- Training Finished ---")

    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        print(traceback.format_exc())
        return # Stop if training fails

    # --- 3. Evaluate Model (on the provided eval_df) ---
    # Note: This evaluates on the same eval_df used during training,
    # which might be the validation split or the original test set.
    if eval_df is not None and not eval_df.empty:
        print("\n--- Evaluating Model on Provided Eval Set ---")
        try:
            if num_labels > 2:
                result, model_outputs, wrong_predictions = model.eval_model(
                    eval_df,
                    report=classification_report, # Pass function for detailed report
                    acc=lambda y_true, y_pred: accuracy_score(y_true, y_pred, average='weighted'),
                    f1=lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
                )
            else:
                result, model_outputs, wrong_predictions = model.eval_model(
                    eval_df,
                    report=classification_report, # Pass function for detailed report
                    acc=accuracy_score,
                    f1=f1_score
                )

            # Before printing or saving as JSON
            result = {k: v.item() if isinstance(v, np.int64) else v for k, v in result.items()} 
       

            print("Evaluation Results:")
            print(json.dumps(result, indent=4))

            # Save evaluation results
            results_path = os.path.join(config.MODEL_ARGS["output_dir"], "eval_results.json")
            with open(results_path, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"Evaluation results saved to {results_path}")

        except Exception as e:
            print(f"Error during final model evaluation: {e}")
            import traceback
            print(traceback.format_exc())
    else:
        print("\nSkipping final evaluation as no eval_df was provided to the train function.")

    print("\n--- Training Function Finished ---")
    return result


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Main Execution ---")

    # --- 1. Set Random Seed ---
    set_seed(config.RANDOM_SEED)

    # --- 2. Load Data ---
    print("\n--- Loading Data ---")
    dataset = Dataset(name=config.DATASET_NAME, config=config.DATASET_CONFIG)
    if not dataset.load():
        print(f"Error: Failed to load dataset. Error: {dataset.error}")
        exit() # Exit if data loading fails

    if dataset.train_df is None:
         print("Error: Training data is None after loading.")
         exit()
    if dataset.num_labels is None:
         print("Error: Number of labels not determined during data loading.")
         exit()


    print(f"Loaded training data: {len(dataset.train_df)} samples.")
    if dataset.test_df is not None:
        print(f"Loaded testing data: {len(dataset.test_df)} samples.")
    else:
        print("No testing data loaded.")

    # Assign to variables for clarity
    initial_train_df = dataset.train_df
    eval_df_prepared = dataset.test_df # Start with the original test set
    num_labels = dataset.num_labels

    # --- 3. Prepare Evaluation Set (if needed) ---
    # If simpletransformers 'evaluate_during_training' is True, it needs an eval set.
    # If no test_df is provided via config, create a validation split from training data.
    if eval_df_prepared is None and config.MODEL_ARGS.get("evaluate_during_training", False):
        print("\n--- Preparing Validation Set ---")
        print("Warning: No test/eval dataset loaded, but evaluate_during_training is True.")
        if len(initial_train_df) > 10: # Only split if training data is reasonably large
            print("Creating validation set from training data (10% split).")
            try:
                # Stratify if possible
                stratify_col = initial_train_df['labels'] if 'labels' in initial_train_df.columns else None
                train_df_split, eval_df_prepared = train_test_split(
                    initial_train_df,
                    test_size=0.1,
                    random_state=config.RANDOM_SEED,
                    stratify=stratify_col
                )
                # Update the initial_train_df to the smaller split part
                initial_train_df = train_df_split
                print(f"New training size (before downsampling): {len(initial_train_df)}, Validation size: {len(eval_df_prepared)}")
            except Exception as e:
                print(f"Could not create validation split: {e}. Disabling evaluate_during_training.")
                config.MODEL_ARGS["evaluate_during_training"] = False # Modify config in-memory
                eval_df_prepared = None # Ensure eval_df is None
        else:
            print("Training data too small to create validation split. Disabling evaluate_during_training.")
            config.MODEL_ARGS["evaluate_during_training"] = False # Modify config in-memory
            eval_df_prepared = None # Ensure eval_df is None


    # --- 4. Apply Downsampling (to training data only) ---
    print("\n--- Applying Downsampling ---")
    train_df_final = apply_downsampling(
        data=initial_train_df, # Use potentially split training data
        method=config.DOWNSAMPLING_METHOD,
        target_size=config.TARGET_SAMPLE_SIZE,
        random_seed=config.RANDOM_SEED,
        label_col='labels' # Assumes label column is 'labels' after Dataset loading
    )

    if len(train_df_final) < len(initial_train_df):
        print(f"Training data downsampled from {len(initial_train_df)} to {len(train_df_final)} samples.")
    else:
        print(f"Training data size remains {len(train_df_final)} (no downsampling applied or target >= original).")


    # --- 5. Call the Training Function ---
    train(
        train_df=train_df_final,
        num_labels=num_labels,
        eval_df=eval_df_prepared # Pass the prepared test/validation set
    )

    print("\n--- Main Execution Finished ---")


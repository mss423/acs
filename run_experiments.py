import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import config
from data_loader import Dataset
from downsampling import apply_downsampling
from train import train, set_seed

def run_all_experiments(methods=None, fractions=None, seeds=None):
    print("--- Starting Batch Experiments for SST2 ---")
    
    # 1. Load Data
    print("\n[1] Loading Data...")
    dataset = Dataset(name="sst2", config=config.DATASET_CONFIG)
    if not dataset.load():
        print(f"Error: Failed to load dataset. Error: {dataset.error}")
        return

    initial_train_df = dataset.train_df
    # We use the test set from the dataset loader as the evaluation set
    eval_df = dataset.test_df
    num_labels = dataset.num_labels
    
    print(f"Loaded {len(initial_train_df)} training samples.")
    print(f"Loaded {len(eval_df)} evaluation samples.")

    # 2. Define Experiment Grid
    if methods is None:
        methods = ['random', 'kmeans', 'dedup', 'acs']
    if fractions is None:
        fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if seeds is None:
        seeds = [42, 43, 44] # 3 Replications
    
    results = []
    
    # Ensure results directory exists
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = config.RESULTS_DIR / f"sst2_experiments_results_{timestamp}.csv"
    
    total_experiments = len(methods) * len(fractions) * len(seeds)
    current_exp = 0

    # 3. Experiment Loop
    for method in methods:
        for frac in fractions:
            # Calculate target size
            target_size = int(len(initial_train_df) * frac)
            if target_size == 0:
                target_size = 1 # Minimum 1 sample
            
            for seed in seeds:
                current_exp += 1
                print(f"\n=== Experiment {current_exp}/{total_experiments} ===")
                print(f"Method: {method}, Fraction: {frac} (k={target_size}), Seed: {seed}")
                
                # Set seed for reproducibility of this run
                set_seed(seed)
                
                try:
                    # Downsample
                    # Note: We pass dataset_name="sst2" to enable embedding caching
                    train_df_downsampled = apply_downsampling(
                        data=initial_train_df,
                        method=method,
                        target_size=target_size,
                        random_seed=seed,
                        dataset_name="sst2" 
                    )
                    
                    # Train & Evaluate
                    # We disable intermediate evaluation to save time, only eval at end
                    # But src/train.py does evaluation if eval_df is passed.
                    # We'll rely on train() returning the result dict.
                    
                    # Temporarily override config to ensure we don't save every model checkpoint to save space
                    # and maybe disable evaluate_during_training to speed up if we just want final result
                    # But let's stick to default behavior or minimal overrides.
                    
                    # We pass num_train_epochs=3 (default)
                    eval_result = train(
                        train_df=train_df_downsampled,
                        num_labels=num_labels,
                        eval_df=eval_df,
                        manual_seed=seed,
                        num_train_epochs=3
                    )
                    
                    # Collect metrics
                    # eval_result is a dict like {'mcc': 0.8, 'eval_loss': 0.5, ...}
                    # We'll add our experiment params
                    row = {
                        'method': method,
                        'fraction': frac,
                        'k_samples': target_size,
                        'seed': seed,
                        'train_size': len(train_df_downsampled)
                    }
                    if eval_result:
                        row.update(eval_result)
                    else:
                        row['error'] = "Training failed or no result"
                    
                    results.append(row)
                    
                    # Save intermediate results
                    pd.DataFrame(results).to_csv(results_file, index=False)
                    print(f"Saved intermediate results to {results_file}")
                    
                except Exception as e:
                    print(f"!!! Error in experiment: {e}")
                    import traceback
                    traceback.print_exc()
                    row = {
                        'method': method,
                        'fraction': frac,
                        'k_samples': target_size,
                        'seed': seed,
                        'error': str(e)
                    }
                    results.append(row)
                    pd.DataFrame(results).to_csv(results_file, index=False)

    print("\n--- All Experiments Completed ---")
    print(f"Final results saved to {results_file}")

if __name__ == "__main__":
    run_all_experiments()

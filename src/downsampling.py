# src/downsampling.py

import pandas as pd
from typing import Optional, Callable, Dict, Any
from utils import *
import config
import os
import pickle
# Optional: Import if your stratified implementation needs it
# from sklearn.model_selection import train_test_split

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Placeholder Sampling Functions ---
# These functions define the expected signature for your custom implementations.
# Replace the body of these functions with your actual downsampling logic.

def sample_random(
    data: pd.DataFrame,
    k_samples: int,
    random_state: Optional[int] = None,
    **kwargs # Allow for extra arguments if needed
) -> pd.DataFrame:
    """
    Performs random sampling on the DataFrame.
    (Placeholder - Replace with your implementation)

    Args:
        data (pd.DataFrame): The input DataFrame.
        k_samples (int): The exact number of samples to return.
        random_state (Optional[int]): Seed for reproducibility.
        **kwargs: Catches unused arguments like label_col.

    Returns:
        pd.DataFrame: The randomly sampled DataFrame.

    Raises:
        NotImplementedError: This is a placeholder.
    """
    print(f"--> Called sample_random with k_samples={k_samples}, random_state={random_state}")
    # --- USER IMPLEMENTATION START ---
    # Example using pandas built-in sampling:
    if k_samples > len(data):
        print(f"Warning: Requested sample size ({k_samples}) is larger than the data size ({len(data)}). Returning original data.")
        return data.copy()
    return data.sample(n=k_samples, random_state=random_state)
    # raise NotImplementedError("Random sampling function needs to be implemented by the user.")
    # --- USER IMPLEMENTATION END ---

def sample_score(
    data: pd.DataFrame,
    k_samples: int,
    **kwargs # Allow for extra arguments if needed
) -> pd.DataFrame:
    """
    Performs score-based sampling on the DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        k_samples (int): The exact number of samples to return.
        **kwargs: Catches unused arguments like label_col.

    Returns:
        pd.DataFrame: The randomly sampled DataFrame.

    Raises:
        NotImplementedError: This is a placeholder.
    """
    dataset = kwargs.get('dataset', 'sst2')
    score_method = kwargs.get('score_method', 'aum')
    print(f"--> Called sample_score with k_samples={k_samples}, score_method={score_method}")
    if k_samples > len(data):
        print(f"Warning: Requested sample size ({k_samples}) is larger than the data size ({len(data)}). Returning original data.")
        return data.copy()

    try:
        score_file = os.path.join(config.PROCESSED_DATA_DIR, f"{dataset}/{score_method}_{dataset}.pkl")
        with open(score_file, "rb") as f:
            score_idx = pickle.load(f)
        return data.iloc[score_idx[:k_samples]]
    except NotImplementedError:
        print(f"No such score file found for {score_method}...")
        raise


def sample_acs(
    data: pd.DataFrame,
    k_samples: int,
    **kwargs # Allow for extra arguments if needed
) -> pd.DataFrame:
    """
    Performs ACS sampling on the DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        k_samples (int): The exact number of samples to return.
        **kwargs: Catches unused arguments like label_col.
         TODO unpack kwargs to get coverage / sim thresh

    Returns:
        pd.DataFrame: The randomly sampled DataFrame.

    Raises:
        NotImplementedError: This is a placeholder.
    """
     # --- Extract ACS specific parameters with defaults ---
    coverage = kwargs.get('coverage', 0.9)
    sim_lb = kwargs.get('sim_lb', 0)
    max_degree = kwargs.get('max_degree', int((5 * len(data) * coverage) / k_samples))
    isCrossner = kwargs.get('isCrossner', False)

    print(f"--> Called sample_acs with k_samples={k_samples}")
    print(f"    ACS Parameters: coverage={coverage}, sim_lb={sim_lb}, max_degree={max_degree}")

    if k_samples > len(data):
        print(f"Warning: Requested sample size ({k_samples}) is larger than the data size ({len(data)}). Returning original data.")
        return data.copy()

    embed_data = kwargs.get('embed_data', None)
    try:
        cos_sim = cosine_similarity(embed_data)

    except Exception as e:
        print(f"No embedding data found, recomputing...")
        cos_sim = cosine_similarity(get_embeddings_task(data['sentence']))

    if isCrossner:
        selected_samples = get_acs_k(cos_sim, None, k_samples, max_degree=max_degree, sim_lb=sim_lb, coverage=coverage)
    else:
        selected_samples = get_acs_k(cos_sim, data['label'], k_samples, max_degree=max_degree, sim_lb=sim_lb, coverage=coverage)
    return data.iloc[selected_samples]


# --- Mapping from method names to functions ---
# The user's custom functions should be added or replace placeholders here.
SAMPLING_FUNCTIONS: Dict[str, Callable[..., pd.DataFrame]] = {
    "random": sample_random,
    "acs": sample_acs,
    "score": sample_score
    # "custom": sample_custom_method,
}


# --- Main Dispatcher Function ---
def apply_downsampling(
    data: pd.DataFrame,
    method: str,
    target_size: Optional[int] = None,
    random_seed: Optional[int] = None,
    label_col: str = 'labels', # Default label column for stratified
    **sampling_args
) -> pd.DataFrame:
    """
    Applies a specified downsampling method to the DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame (expected to have 'text' and 'labels').
        method (str): The name of the downsampling method to use
                      (e.g., 'random', 'stratified', 'none'). Must be a key in
                      SAMPLING_FUNCTIONS or 'none'.
        target_size (Optional[int]): The desired number of samples after downsampling.
                                     Required unless method is 'none'.
        random_seed (Optional[int], optional): Seed for reproducibility. Defaults to None.
        label_col (str, optional): The column name containing labels, primarily
                                   used for 'stratified' sampling. Defaults to 'labels'.

    Returns:
        pd.DataFrame: The downsampled DataFrame, or the original DataFrame if
                      method is 'none' or target_size is invalid.

    Raises:
        ValueError: If the specified method is unknown or if target_size is
                    missing when required.
    """
    print(f"\nApplying downsampling:")
    print(f"  Method: {method}")
    print(f"  Target Size: {target_size}")
    print(f"  Original Data Size: {len(data)}")
    print(f"  Random Seed: {random_seed}")
    if method == 'stratified':
        print(f"  Label Column: {label_col}")

    if method.lower() == 'none':
        print("  Method is 'none', returning original data.")
        return data.copy() # Return a copy to avoid modifying original df later

    if not isinstance(target_size, int) or target_size <= 0:
        raise ValueError("A positive integer 'target_size' is required for downsampling methods.")

    if target_size >= len(data):
        print(f"  Warning: Target size ({target_size}) >= original size ({len(data)}). Returning original data.")
        return data.copy()

    if method not in SAMPLING_FUNCTIONS:
        raise ValueError(f"Unknown downsampling method: '{method}'. Available methods: {list(SAMPLING_FUNCTIONS.keys())}")

    # Get the appropriate sampling function
    sampling_func = SAMPLING_FUNCTIONS[method]

    args_to_pass = {
        'data': data,
        'k_samples': target_size,
        'random_state': random_seed,
        'label_col': label_col,
        **sampling_args
    }

    # Call the sampling function
    try:
        downsampled_df = sampling_func(**args_to_pass)
        print(f"Downsampling complete. Resulting size: {len(downsampled_df)}")
        return downsampled_df
    except NotImplementedError:
        print(f"Error: The implementation for the '{method}' sampling method is missing.")
        raise # Re-raise the error
    except Exception as e:
        print(f"Error during '{method}' downsampling: {e}")
        # Depending on desired behavior, you could return original data or raise
        # raise e # Re-raise the specific error (e.g., ValueError from stratification)
        print("  Returning original data due to error during downsampling.")
        return data.copy()


# --- Example Usage ---
if __name__ == '__main__':
    # Create dummy data
    dummy_data = {
        'text': [f'Sentence {i}' for i in range(20)],
        'labels': ([0] * 10) + ([1] * 10) # Balanced labels
    }
    dummy_df = pd.DataFrame(dummy_data)
    print("--- Original Dummy DataFrame ---")
    print(dummy_df)
    print(f"Label distribution:\n{dummy_df['labels'].value_counts()}")

    # --- Test Case 1: Random Sampling ---
    print("\n--- Testing Random Sampling (target_size=5) ---")
    try:
        # NOTE: Using the example implementation provided in sample_random placeholder
        random_sample = apply_downsampling(dummy_df, method='random', target_size=5, random_seed=42)
        print("Result:")
        print(random_sample)
    except Exception as e:
        print(f"Error during random sampling test: {e}")

    # --- Test Case 2: Stratified Sampling ---
    print("\n--- Testing Stratified Sampling (target_size=8) ---")
    try:
        # NOTE: Using the example implementation provided in sample_stratified placeholder
        stratified_sample = apply_downsampling(dummy_df, method='stratified', target_size=8, random_seed=123, label_col='labels')
        print("Result:")
        print(stratified_sample)
        print(f"Result label distribution:\n{stratified_sample['labels'].value_counts()}")
    except Exception as e:
        print(f"Error during stratified sampling test: {e}")


    # --- Test Case 3: Method 'none' ---
    print("\n--- Testing Method 'none' ---")
    try:
        none_sample = apply_downsampling(dummy_df, method='none')
        print("Result:")
        print(none_sample)
        assert len(none_sample) == len(dummy_df), "Method 'none' should return original size"
    except Exception as e:
        print(f"Error during 'none' sampling test: {e}")


    # --- Test Case 4: Target size too large ---
    print("\n--- Testing Target Size Too Large (target_size=30) ---")
    try:
        large_sample = apply_downsampling(dummy_df, method='random', target_size=30)
        print("Result:")
        print(large_sample)
        assert len(large_sample) == len(dummy_df), "Target size >= original should return original size"
    except Exception as e:
        print(f"Error during large target size test: {e}")

    # --- Test Case 5: Unknown method ---
    print("\n--- Testing Unknown Method ---")
    try:
        unknown_sample = apply_downsampling(dummy_df, method='unknown_method', target_size=5)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

    # --- Test Case 6: Missing target size ---
    print("\n--- Testing Missing Target Size ---")
    try:
        missing_size_sample = apply_downsampling(dummy_df, method='random')
    except ValueError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

    # --- Test Case 7: Stratified sampling requires scikit-learn (if placeholder uses it) ---
    # (This test assumes the placeholder uses sklearn and it might not be installed)
    # print("\n--- Testing Stratified Sampling without scikit-learn (Illustrative) ---")
    # # You would need to temporarily uninstall sklearn to truly test this,
    # # or modify the placeholder to check for its existence differently.

#!/usr/bin/env python3
"""
Split pre_deployment_data.pkl into train and test sets
Uses the same splitting logic as ml_pipelines.py
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path

def load_data(file_path='pre_deployment_data.pkl'):
    """Load and prepare the data"""
    print("Loading data...")
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    merged = data_dict['merged_df']
    HRV_features = data_dict['HRV_features']
    EMG_features = data_dict['EMG_features']
    CUT_features = data_dict['CUT_feature']
    SPSS_features = data_dict['SPSS_feature']
    categorical_features = data_dict['categorical_features']
    PTSD_features = data_dict['PTSD_features']
    
    # Combine all numerical features
    all_numeric_features = (HRV_features + CUT_features + 
                           SPSS_features + EMG_features)
    
    print(f"Total numerical features: {len(all_numeric_features)}")
    
    return {
        'merged_df': merged,
        'HRV_features': HRV_features,
        'EMG_features': EMG_features,
        'CUT_feature': CUT_features,
        'SPSS_feature': SPSS_features,
        'categorical_features': categorical_features,
        'PTSD_features': PTSD_features,
        'numeric_features': all_numeric_features
    }

def simple_handle_missing_values(df, numeric_features):
    """Simple missing value handling: remove any sample with NA in numeric features"""
    print("Handling missing values with simple NA removal...")
    
    # Create a copy to work with
    df_clean = df.copy()
    
    # Convert numeric features to numeric and get missing stats
    for col in numeric_features:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Count missing values before removal
    initial_count = len(df_clean)
    missing_counts = df_clean[numeric_features].isna().sum()
    total_missing = missing_counts.sum()
    
    print(f"Initial dataset size: {initial_count}")
    print(f"Total missing values in numeric features: {total_missing}")
    
    if total_missing > 0:
        print(f"Features with missing values:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  {col}: {count} ({count/initial_count*100:.1f}%)")
    
    # Remove rows with any missing numeric values
    df_clean = df_clean.dropna(subset=numeric_features)
    
    final_count = len(df_clean)
    removed_count = initial_count - final_count
    
    print(f"Removed {removed_count} samples with missing values")
    print(f"Final dataset size: {final_count}")
    print(f"Retention rate: {final_count/initial_count*100:.1f}%")
    
    return df_clean

def custom_train_test_split(X, y, test_size=0.15, min_positive_test_ratio=0.4, random_state=42):
    """Custom split ensuring sufficient positive samples in test set"""
    print(f"Performing custom train-test split with {min_positive_test_ratio*100}% of positive cases in test...")
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Get indices of positive and negative cases
    positive_indices = np.where(y == 1)[0]
    negative_indices = np.where(y == 0)[0]
    
    n_positive = len(positive_indices)
    n_negative = len(negative_indices)
    
    print(f"Original data: {len(y)} samples ({n_positive} positive, {n_negative} negative)")
    
    # Calculate splits
    n_positive_test = max(int(n_positive * min_positive_test_ratio), 5)  # At least 5 positive samples
    total_test_size = int(len(y) * test_size)
    n_negative_test = min(total_test_size - n_positive_test, n_negative)
    
    # Randomly sample test indices
    test_positive_indices = np.random.choice(positive_indices, size=n_positive_test, replace=False)
    test_negative_indices = np.random.choice(negative_indices, size=n_negative_test, replace=False)
    
    # Combine test indices
    test_indices = np.concatenate([test_positive_indices, test_negative_indices])
    train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
    
    # Create train and test sets
    X_train = X.iloc[train_indices] if hasattr(X, 'iloc') else X[train_indices]
    X_test = X.iloc[test_indices] if hasattr(X, 'iloc') else X[test_indices]
    y_train = y.iloc[train_indices] if hasattr(y, 'iloc') else y[train_indices]
    y_test = y.iloc[test_indices] if hasattr(y, 'iloc') else y[test_indices]
    
    print(f"Train set: {len(X_train)} samples, {sum(y_train)} positive ({sum(y_train)/len(y_train)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples, {sum(y_test)} positive ({sum(y_test)/len(y_test)*100:.1f}%)")
    print(f"Positive cases in test: {sum(y_test)}/{n_positive} ({sum(y_test)/n_positive*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test, train_indices, test_indices

def split_and_save_data(input_file='pre_deployment_data.pkl', 
                       output_train='pre_deployment_data_train.pkl',
                       output_test='pre_deployment_data_test.pkl',
                       test_size=0.15,
                       min_positive_test_ratio=0.4,
                       random_state=42):
    """
    Split the data and save train/test sets as separate pickle files
    
    Args:
        input_file: Path to original pickle file
        output_train: Path for training set pickle file
        output_test: Path for test set pickle file
        test_size: Proportion of data for test set
        min_positive_test_ratio: Minimum proportion of positive cases in test
        random_state: Random seed for reproducibility
    """
    
    print("="*70)
    print("SPLITTING PRE_DEPLOYMENT_DATA.PKL INTO TRAIN/TEST SETS")
    print("="*70)
    
    # Load original data
    data_dict = load_data(input_file)
    merged_df = data_dict['merged_df']
    numeric_features = data_dict['numeric_features']
    
    # Handle missing values
    merged_clean = simple_handle_missing_values(merged_df, numeric_features)
    
    # Prepare target variable only
    y = merged_clean['CAPSF1I2s.0'].copy()
    
    # Remove missing targets (if any remain)
    mask = ~y.isna()
    y = y[mask]
    merged_clean = merged_clean[mask]
    
    y = y.astype(int)
    
    print(f"Final dataset shape: {merged_clean.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    print(f"Class ratio: {len(y[y==1])/len(y)*100:.1f}% positive")
    
    # Perform train-test split using indices
    _, _, _, _, train_indices, test_indices = custom_train_test_split(
        merged_clean, y, test_size=test_size, min_positive_test_ratio=min_positive_test_ratio, random_state=random_state
    )
    
    # Create train and test dataframes with all original columns
    train_df = merged_clean.iloc[train_indices].copy()
    test_df = merged_clean.iloc[test_indices].copy()
    
    # Create train data dictionary with all original features
    train_data_dict = {
        'merged_df': train_df,
        'HRV_features': data_dict['HRV_features'],
        'EMG_features': data_dict['EMG_features'],
        'CUT_feature': data_dict['CUT_feature'],
        'SPSS_feature': data_dict['SPSS_feature'],
        'categorical_features': data_dict['categorical_features'],
        'PTSD_features': data_dict['PTSD_features'],
        'numeric_features': data_dict['numeric_features']
    }
    
    # Create test data dictionary with all original features
    test_data_dict = {
        'merged_df': test_df,
        'HRV_features': data_dict['HRV_features'],
        'EMG_features': data_dict['EMG_features'],
        'CUT_feature': data_dict['CUT_feature'],
        'SPSS_feature': data_dict['SPSS_feature'],
        'categorical_features': data_dict['categorical_features'],
        'PTSD_features': data_dict['PTSD_features'],
        'numeric_features': data_dict['numeric_features']
    }
    
    # Save train set
    print(f"\nSaving training set to {output_train}...")
    with open(output_train, 'wb') as f:
        pickle.dump(train_data_dict, f)
    print(f"Training set saved: {len(train_df)} samples")
    
    # Save test set
    print(f"Saving test set to {output_test}...")
    with open(output_test, 'wb') as f:
        pickle.dump(test_data_dict, f)
    print(f"Test set saved: {len(test_df)} samples")
    
    # Print summary
    print("\n" + "="*70)
    print("DATA SPLITTING COMPLETE")
    print("="*70)
    print(f"Original dataset: {len(merged_clean)} samples")
    print(f"Training set: {len(train_df)} samples ({len(train_df)/len(merged_clean)*100:.1f}%)")
    print(f"Test set: {len(test_df)} samples ({len(test_df)/len(merged_clean)*100:.1f}%)")
    print(f"Files created:")
    print(f"  - {output_train}")
    print(f"  - {output_test}")
    
    return train_data_dict, test_data_dict

def verify_split(train_file, test_file):
    """Verify the train/test split files"""
    print("\n" + "="*70)
    print("VERIFYING SPLIT FILES")
    print("="*70)
    
    # Load train set
    with open(train_file, 'rb') as f:
        train_dict = pickle.load(f)
    
    # Load test set
    with open(test_file, 'rb') as f:
        test_dict = pickle.load(f)
    
    train_df = train_dict['merged_df']
    test_df = test_dict['merged_df']
    
    print(f"Training set:")
    print(f"  Shape: {train_df.shape}")
    print(f"  PTSD distribution: {train_df['CAPSF1I2s.0'].value_counts().to_dict()}")
    print(f"  Numeric features: {len(train_dict['numeric_features'])}")
    
    print(f"\nTest set:")
    print(f"  Shape: {test_df.shape}")
    print(f"  PTSD distribution: {test_df['CAPSF1I2s.0'].value_counts().to_dict()}")
    print(f"  Numeric features: {len(test_dict['numeric_features'])}")
    
    # Check for overlap
    train_indices = set(train_df.index)
    test_indices = set(test_df.index)
    overlap = train_indices & test_indices
    
    if len(overlap) == 0:
        print(f"\n✓ No overlap between train and test sets")
    else:
        print(f"\n✗ WARNING: {len(overlap)} samples overlap between train and test!")
    
    print("Verification complete.")

if __name__ == "__main__":
    # Configuration
    config = {
        'input_file': 'pre_deployment_data.pkl',
        'output_train': 'pre_deployment_data_train.pkl',
        'output_test': 'pre_deployment_data_test.pkl',
        'test_size': 0.15,
        'min_positive_test_ratio': 0.4,
        'random_state': 42
    }
    
    # Check if input file exists
    if not os.path.exists(config['input_file']):
        print(f"Error: Input file '{config['input_file']}' not found!")
        exit(1)
    
    # Split the data
    train_dict, test_dict = split_and_save_data(**config)
    
    # Verify the split
    verify_split(config['output_train'], config['output_test'])
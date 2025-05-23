import pandas as pd
import numpy as np
import joblib
import os

def load_dataset(file_path):
    """
    Load the email dataset from a CSV file.
    Assumes the first column is an identifier and the last column is the label (spam=1, ham=0).
    """
    try:
        data = pd.read_csv(file_path, encoding='latin-1')
        print("Dataset loaded successfully!")
        return data
    except FileNotFoundError:
        print("Error: Dataset file not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_data(data):
    """
    Preprocess the dataset: remove identifier, handle missing values, and extract features/labels.
    """
    # Drop the 'Email No.' column (identifier)
    if 'Email No.' in data.columns:
        data = data.drop(columns=['Email No.'])
    
    # Assume the last column is the label (adjust if label column has a specific name)
    X = data.iloc[:, :-1].values  # All columns except the last one
    y = data.iloc[:, -1].values   # Last column as labels
    
    # Handle missing values (replace NaN with 0 for word counts)
    X = np.nan_to_num(X, nan=0.0)
    
    # Ensure labels are binary (0 or 1)
    y = np.where(y > 0, 1, 0)
    
    return X, y

def save_processed_data(X, y, output_path):
    """
    Save preprocessed data to a CSV file (optional).
    """
    df = pd.DataFrame(X, columns=[f'word_{i}' for i in range(X.shape[1])])
    df['label'] = y
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

def main():
    # Paths
    data_path = 'data/raw/emails.csv'
    processed_path = 'data/processed/processed_data.csv'
    
    # Create directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Load dataset
    data = load_dataset(data_path)
    if data is None:
        return
    
    # Preprocess data
    print("Preprocessing data...")
    X, y = preprocess_data(data)
    
    # Save preprocessed data (optional)
    save_processed_data(X, y, processed_path)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import joblib
import os
import sys
import re
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, save_npz, load_npz  # Add these imports


# Download required NLTK data
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()



# def load_dataset(file_path):
#     """
#     Load the email dataset from a CSV file.
#     Assumes the first column is an identifier and the last column is the label (spam=1, ham=0).
#     """
#     try:
#         data = pd.read_csv(file_path, encoding='latin-1')
#         print("Dataset loaded successfully!")
#         return data
#     except FileNotFoundError:
#         print("Error: Dataset file not found. Please check the file path.")
#         return None
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         return None

# def preprocess_data(data):
#     """
#     Preprocess the dataset: remove identifier, handle missing values, and extract features/labels.
#     """
#     # Drop the 'Email No.' column (identifier)
#     if 'Email No.' in data.columns:
#         data = data.drop(columns=['Email No.'])
    
#     # Assume the last column is the label (adjust if label column has a specific name)
#     X = data.iloc[:, :-1].values  # All columns except the last one
#     y = data.iloc[:, -1].values   # Last column as labels
    
#     # Handle missing values (replace NaN with 0 for word counts)
#     X = np.nan_to_num(X, nan=0.0)
    
#     # Ensure labels are binary (0 or 1)
#     y = np.where(y > 0, 1, 0)
    
#     return X, y

# def save_processed_data(X, y, output_path):
#     """
#     Save preprocessed data to a CSV file (optional).
#     """
#     df = pd.DataFrame(X, columns=[f'word_{i}' for i in range(X.shape[1])])
#     df['label'] = y
#     df.to_csv(output_path, index=False)
#     print(f"Preprocessed data saved to {output_path}")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


import pandas as pd
import numpy as np
import joblib
import os
import sys
import re
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, save_npz, load_npz  # Add these imports

# ...existing code...

def preprocess_data(data):
    """
    Preprocess the dataset with only label and text columns.
    Assumes first column is label (spam/ham) and second column is email text.
    """
    from utils.email_processor import process_email, load_vocabulary
    
    # Convert labels from text to binary (ham=0, spam=1)
    y = np.where(data.iloc[:, 0].str.lower() == 'spam', 1, 0)
    
    # Get email text and handle missing values
    texts = data.iloc[:, 1].fillna('').values
    print(f"Total emails: {len(texts)}")
    print(f"Missing texts: {sum(pd.isna(data.iloc[:, 1]))}")
 
    # Load or create vocabulary from training data with progress bar
    print("Building vocabulary...")
    all_words = set()
    for text in tqdm(texts, desc="Building vocab", unit="email"):
        if pd.isna(text) or not isinstance(text, str):
            continue
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        words = [stemmer.stem(word) for word in words if word not in stop_words]
        all_words.update(words)
    
    vocab = sorted(list(all_words))
    print(f"Vocabulary size: {len(vocab)} words")
    
    # Convert each text to feature vector using sparse matrix
    print("\nConverting emails to feature vectors...")
    rows = []
    cols = []
    data = []
    
    for i, text in tqdm(enumerate(texts), total=len(texts), 
                       desc="Processing", unit="email"):
        if pd.isna(text) or not isinstance(text, str):
            continue
        # Get word counts for this text
        word_counts = process_email(text, vocab)[0]
        # Only store non-zero entries
        nonzero_idx = np.nonzero(word_counts)[0]
        rows.extend([i] * len(nonzero_idx))
        cols.extend(nonzero_idx)
        data.extend(word_counts[nonzero_idx])
    
    X = csr_matrix((data, (rows, cols)), shape=(len(texts), len(vocab)))
    return X, y

def save_processed_data(X, y, output_path):
    """
    Save preprocessed sparse data to files.
    """
    # Save sparse matrix
    sparse_path = output_path.replace('.csv', '_sparse.npz')
    save_npz(sparse_path, X)
    
    # Save labels separately
    labels_path = output_path.replace('.csv', '_labels.npy')
    np.save(labels_path, y)
    
    print(f"Preprocessed data saved to {sparse_path} and {labels_path}")

def load_dataset(file_path):
    """
    Load the email dataset with label and text columns.
    """
    try:
        # Read only first 2 columns - label and text
        data = pd.read_csv(file_path, encoding='latin-1', usecols=[0,1])
        print("Dataset loaded successfully!")
        print(f"Number of samples: {len(data)}")
        return data
    except FileNotFoundError:
        print("Error: Dataset file not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None



def main():
    # Paths
    data_path = os.path.join(project_root, 'data', 'raw', 'spam_Emails_data.csv')
    processed_path = os.path.join(project_root, 'data', 'processed', 'processed_data.csv')
    
    # Create directories if they don't exist
    os.makedirs(os.path.join(project_root, 'data', 'raw'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'data', 'processed'), exist_ok=True)
    
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
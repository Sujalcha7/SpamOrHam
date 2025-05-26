import pandas as pd
import numpy as np
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download NLTK data
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def process_email(email_text, vocab):
    """
    Convert email text into a feature vector based on the dataset's vocabulary.
    """
    # Tokenize the email: lowercase, remove punctuation, split into words
    email_text = email_text.lower()
    email_text = re.sub(r'[^\w\s]', '', email_text)  # Remove punctuation
    words = email_text.split()
    
    # Remove stopwords and apply stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    # Count word frequencies
    word_counts = Counter(words)
    
    # Create feature vector: count of each vocab word in the email
    feature_vector = np.zeros(len(vocab))
    for i, word in enumerate(vocab):
        feature_vector[i] = word_counts.get(word, 0)
    
    return feature_vector.reshape(1, -1)  # Reshape for model prediction

def get_top_features(email_text, vocab, top_n=5):
    """
    Return the top N words with their frequencies for debugging.
    """
    email_text = email_text.lower()
    email_text = re.sub(r'[^\w\s]', '', email_text)
    words = email_text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    word_counts = Counter(words)
    
    # Filter to words in vocab
    vocab_counts = {word: count for word, count in word_counts.items() if word in vocab}
    # Sort by frequency
    top_features = sorted(vocab_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_features


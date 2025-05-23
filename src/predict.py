import numpy as np
import joblib
from data_preprocessing import load_dataset, preprocess_data

def load_model(model_path):
    """
    Load the trained SVM model.
    """
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully!")
        return model
    except FileNotFoundError:
        print("Error: Model file not found.")
        return None

def predict_spam(model, X):
    """
    Predict whether emails are spam (1) or ham (0).
    """
    predictions = model.predict(X)
    return predictions

def main():
    # Paths
    data_path = 'data/raw/emails.csv'
    model_path = 'models/svm_spam_model.pkl'
    
    # Load model
    model = load_model(model_path)
    if model is None:
        return
    
    # Load and preprocess data (for testing predictions)
    data = load_dataset(data_path)
    if data is None:
        return
    
    X, y = preprocess_data(data)
    
    # Predict on a sample (first 5 emails for demonstration)
    sample_X = X[:5]
    sample_y = y[:5]
    predictions = predict_spam(model, sample_X)
    
    print("\nSample Predictions (0=ham, 1=spam):")
    for i, (pred, true) in enumerate(zip(predictions, sample_y)):
        print(f"Email {i+1}: Predicted={pred}, True={true}")

if __name__ == "__main__":
    main()
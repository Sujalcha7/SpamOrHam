import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
from .data_preprocessing import load_dataset, preprocess_data 

def train_svm(X, y):
    """
    Train an SVM model with hyperparameter tuning.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define SVM model
    svm = SVC(kernel='linear')
    
    # Hyperparameter tuning
    param_grid = {'C': [0.1, 1, 10]}
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return best_model, X_test, y_test

def save_model(model, output_path):
    """
    Save the trained SVM model.
    """
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")

def main():
    # Paths
    data_path = 'data/raw/emails.csv'
    model_path = 'models/svm_spam_model.pkl'
    
    # Load and preprocess data
    data = load_dataset(data_path)
    if data is None:
        return
    
    X, y = preprocess_data(data)
    
    # Train model
    print("Training SVM model...")
    model, X_test, y_test = train_svm(X, y)
    
    # Save model
    save_model(model, model_path)

if __name__ == "__main__":
    main()
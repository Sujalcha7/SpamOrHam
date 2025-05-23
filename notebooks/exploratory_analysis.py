import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.data_preprocessing import load_dataset, preprocess_data
from src.model_training import train_svm

# Load dataset
data_path = 'data/raw/emails.csv'
data = load_dataset(data_path)

# Preprocess data
X, y = preprocess_data(data)

# Plot class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title('Class Distribution (0=Ham, 1=Spam)')
plt.xlabel('Label')
plt.ylabel('Count')
plt.savefig('data/processed/class_distribution.png')
plt.show()

# Train model and get test set predictions
model, X_test, y_test = train_svm(X, y)

# Plot confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('data/processed/confusion_matrix.png')
plt.show()
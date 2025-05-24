from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
import joblib
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

from utils.email_processor import process_email, get_top_features

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Add Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vocabulary
model_path = os.path.join(project_root, 'models', 'spam_classifier.pkl')
vocab_path = os.path.join(project_root, 'models', 'vocabulary.pkl')

try:
    model = joblib.load(model_path)
    vocab = joblib.load(vocab_path)
    print("Model and vocabulary loaded successfully!")
except Exception as e:
    print(f"Error loading model or vocabulary: {str(e)}")
    sys.exit(1)

class EmailRequest(BaseModel):
    email_text: str

@app.post("/predict")
async def predict(request: EmailRequest):
    try:
        if not request.email_text:
            raise HTTPException(status_code=400, detail="Email text is required")

        # Process email into feature vector
        feature_vector = process_email(request.email_text, vocab)
        
        # Predict
        prediction = model.predict(feature_vector)[0]
        label = "Spam" if prediction == 1 else "Ham"
        
        # Get confidence score
        decision_score = model.decision_function(feature_vector)[0]
        confidence = float(1 / (1 + np.exp(-decision_score)))
        
        # Get top contributing words
        top_words = get_top_features(request.email_text, vocab, top_n=5)
        
        return {
            'label': label,
            'confidence': confidence,
            'top_words': top_words
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
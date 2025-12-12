# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from typing import List


# TODO: Create FastAPI app instance
app = FastAPI()

# TODO: Load sentiment analysis pipeline
sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
request_count = 0

# Hint: pipeline("sentiment-analysis")

class TextInput(BaseModel):
    text: str

# Add this new model
class BatchTextInput(BaseModel):
    texts: List[str]

class BatchPredictionResult(BaseModel):
    text: str
    sentiment: str
    confidence: float

class BatchResponse(BaseModel):
    results: List[BatchPredictionResult]
    total_processed: int
    summary: dict

@app.post("/predict")
def predict_sentiment(input: TextInput):
    # TODO: Get prediction from classifier
    # TODO: Return result as JSON

    global request_count
    request_count += 1

    text = input.text.strip()

    if(len(text) == 0):
        return {
            "error": "Text cannot be empty"
        } 
    
    if(len(text) < 3):
        return {
            "error": "Enter a larger text"
        }
  
    prediction = sentiment_classifier(text)

    response = {
    "text": text,
    "sentiment": prediction[0]["label"],
    "confidence": prediction[0]["score"]
    }

    return response

# TODO: Add a welcome endpoint at "/"
@app.get("/")
def home():
    return {
        "message": "Welcome to Sentiment API",
        "endpoints": {
            "docs": "/docs",
            "predict": "/predict",
            "batch_predict": "/batch-predict",  # Add this line
            "stats": "/stats"
        }
    }

@app.get("/stats")
def get_stats():
    return {
        "total_predictions": request_count
    }

@app.post("/batch-predict", response_model=BatchResponse)
def predict_batch_sentiment(input: BatchTextInput):
    global request_count
    
    results = []
    sentiments_count = {"POSITIVE": 0, "NEGATIVE": 0}
    total_confidence = 0
    
    for text in input.texts:
        # Clean and validate
        cleaned_text = text.strip()
        
        # Skip invalid texts
        if len(cleaned_text) == 0 or len(cleaned_text) < 3:
            continue
            
        # Predict
        prediction = sentiment_classifier(cleaned_text)
        label = prediction[0]["label"]
        confidence = prediction[0]["score"]
        
        # Add to results
        results.append(
            BatchPredictionResult(
                text=cleaned_text,
                sentiment=label,
                confidence=confidence
            )
        )
        
        # Track statistics
        sentiments_count[label] = sentiments_count.get(label, 0) + 1
        total_confidence += confidence
        request_count += 1
    
    # Calculate summary
    total_processed = len(results)
    avg_confidence = total_confidence / total_processed if total_processed > 0 else 0
    
    summary = {
        "total_texts": len(input.texts),
        "successfully_processed": total_processed,
        "positive_count": sentiments_count.get("POSITIVE", 0),
        "negative_count": sentiments_count.get("NEGATIVE", 0),
        "average_confidence": round(avg_confidence, 4),
        "most_common_sentiment": max(sentiments_count, key=sentiments_count.get) if sentiments_count else "NONE"
    }
    
    return BatchResponse(
        results=results,
        total_processed=total_processed,
        summary=summary
    )
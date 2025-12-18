# main.py
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from transformers import pipeline
from typing import List
import uuid
import json
from redis import Redis
from rq import Queue
from jobs import process_batch_job
import math

# Connect to Redis
try:
    redis_conn = Redis(host='localhost', port=6379, db=0, decode_responses=True)
    task_queue = Queue('default', connection=redis_conn)
    redis_available = True
except Exception as e:
    print(f"Redis connection failed: {e}")
    redis_available = False

# TODO: Create FastAPI app instance
app = FastAPI()

# TODO: Load sentiment analysis pipeline
sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
request_count = 0

#helper function
def apply_temperature_scaling(logits: list, temperature: float = 2.5):

    # Apply temperature
    scaled_logits = [l / temperature for l in logits]
    
    # Softmax to get probabilities
    exp_logits = [math.exp(l) for l in scaled_logits]
    sum_exp = sum(exp_logits)
    probs = [e / sum_exp for e in exp_logits]
    
    return probs


def get_calibrated_confidence(raw_score: float, sentiment: str):

    # Apply non-linear calibration
    # This formula is based on calibration studies for sentiment models
    
    if raw_score >= 0.99:
        # Very high confidence → reduce to reasonable level
        calibrated = 0.75 + (raw_score - 0.99) * 5  # Maps 0.99-1.0 to 0.75-0.80
    elif raw_score >= 0.95:
        # High confidence → moderate reduction
        calibrated = 0.65 + (raw_score - 0.95) * 2.5  # Maps 0.95-0.99 to 0.65-0.75
    elif raw_score >= 0.90:
        # Medium-high confidence → slight reduction
        calibrated = 0.60 + (raw_score - 0.90) * 1  # Maps 0.90-0.95 to 0.60-0.65
    elif raw_score >= 0.80:
        # Medium confidence → minimal adjustment
        calibrated = 0.55 + (raw_score - 0.80) * 0.5  # Maps 0.80-0.90 to 0.55-0.60
    else:
        # Low confidence → keep similar
        calibrated = raw_score * 0.7  # Slight reduction
    
    # Ensure calibrated score doesn't exceed raw score
    calibrated = min(calibrated, raw_score)
    
    # Add small penalty for negative sentiment (they tend to be less reliable)
    if sentiment == "NEGATIVE":
        calibrated *= 0.95
    
    return round(calibrated, 4)

# Hint: pipeline("sentiment-analysis")

class TextInput(BaseModel):
    text: str

# Add this new model
class BatchTextInput(BaseModel):
    texts: List[str]

class PredictionResult(BaseModel):
    text: str
    sentiment: str
    raw_confidence: float
    calibrated_confidence: float
    confidence_level: str  # "high", "medium", "low"

class BatchPredictionResult(BaseModel):
    text: str
    sentiment: str
    raw_confidence: float
    calibrated_confidence: float
    confidence_level: str

class BatchResponse(BaseModel):
    results: List[BatchPredictionResult]
    total_processed: int
    summary: dict

@app.post("/predict", response_model=PredictionResult)
def predict_sentiment(input: TextInput):
    global request_count
    request_count += 1

    text = input.text.strip()

    if len(text) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(text) < 3:
        raise HTTPException(status_code=400, detail="Text must be at least 3 characters long")

    # Get full probability distribution (modern API)
    prediction = sentiment_classifier(text, top_k=None)
    prediction = sentiment_classifier(text, top_k=None)

    # Normalize HF output shape
    if isinstance(prediction[0], dict):
        # HF returned only top label → reconstruct distribution
        scores = prediction
    else:
        # HF returned full distribution
        scores = prediction[0]

    # Extract POSITIVE and NEGATIVE scores
    pos = next(s["score"] for s in scores if s["label"] == "POSITIVE")
    neg = next(s["score"] for s in scores if s["label"] == "NEGATIVE")

    # Raw label and confidence
    label = "POSITIVE" if pos >= neg else "NEGATIVE"
    raw_confidence = max(pos, neg)

    # Calibrated confidence
    calibrated_confidence = get_calibrated_confidence(raw_confidence, label)

    # Neutral logic
    if calibrated_confidence < 0.75 or abs(pos - neg) < 0.2:
        sentiment = "NEUTRAL"
    else:
        sentiment = label

    # Confidence level (based on calibrated confidence)
    if calibrated_confidence >= 0.70:
        confidence_level = "high"
    elif calibrated_confidence >= 0.55:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    return PredictionResult(
        text=text,
        sentiment=sentiment,   # ✅ FIXED
        raw_confidence=round(raw_confidence, 4),
        calibrated_confidence=round(calibrated_confidence, 4),
        confidence_level=confidence_level
    )

# TODO: Add a welcome endpoint at "/"
@app.get("/")
def home():
    return {
        "message": "Welcome to Sentiment API",
        "endpoints": {
            "docs": "/docs",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "batch_predict_async": "/batch-predict-async",  # New
            "job_status": "/job/{job_id}/status",  # New
            "job_results": "/job/{job_id}/results",  # New
            "stats": "/stats"
        },
        "async_available": redis_available
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

        # Determine final sentiment (POSITIVE / NEGATIVE / NEUTRAL)
        if calibrated_confidence < 0.75 or abs(pos - neg) < 0.2:
            sentiment = "NEUTRAL"
        else:
            sentiment = label

        # Add to results
        results.append(
            BatchPredictionResult(
                text=cleaned_text,
                sentiment=sentiment,  # POSITIVE / NEGATIVE / NEUTRAL
                raw_confidence=round(raw_confidence, 4),
                calibrated_confidence=calibrated_confidence,
                confidence_level=confidence_level
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
        "neutral_count": sentiments_count.get("NEUTRAL", 0),
        "average_confidence": round(avg_calibrated_confidence, 4),
        "most_common_sentiment": max(sentiments_count, key=sentiments_count.get)
    }

    
    return BatchResponse(
        results=results,
        total_processed=total_processed,
        summary=summary
    )

# ==================== ASYNC BATCH ENDPOINTS ====================

class AsyncBatchResponse(BaseModel):
    job_id: str
    status: str
    message: str

@app.post("/batch-predict-async", response_model=AsyncBatchResponse)
def submit_batch_job_async(input: BatchTextInput):
    """
    Submit a batch prediction job for async processing.
    Returns immediately with a job_id.
    """
    if not redis_available:
        return {
            "job_id": None,
            "status": "error",
            "message": "Redis not available. Use /batch-predict for synchronous processing."
        }
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job in Redis
    redis_conn.hset(f"job:{job_id}", mapping={
        "status": "queued",
        "total": len(input.texts),
        "processed": 0,
        "percent": 0
    })
    
    # Enqueue job for background processing
    task_queue.enqueue(
        process_batch_job,
        job_id,
        input.texts,
        job_timeout='10m'  # 10 minute timeout
    )
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Job submitted successfully. Total texts: {len(input.texts)}"
    }


@app.get("/job/{job_id}/status")
def get_job_status(job_id: str):
    """
    Check the status of an async batch job.
    """
    if not redis_available:
        return {"error": "Redis not available"}
    
    # Check if job exists
    if not redis_conn.exists(f"job:{job_id}"):
        return {"error": "Job not found"}
    
    # Get job data
    job_data = redis_conn.hgetall(f"job:{job_id}")
    
    status = job_data.get("status", "unknown")
    total = int(job_data.get("total", 0))
    processed = int(job_data.get("processed", 0))
    percent = int(job_data.get("percent", 0))
    
    response = {
        "job_id": job_id,
        "status": status,
        "progress": f"{processed}/{total}",
        "percent": percent
    }
    
    # If completed, include summary
    if status == "completed" and "summary" in job_data:
        response["summary"] = json.loads(job_data["summary"])
    
    return response


@app.get("/job/{job_id}/results")
def get_job_results(job_id: str):
    """
    Get the results of a completed batch job.
    """
    if not redis_available:
        return {"error": "Redis not available"}
    
    # Check if job exists
    if not redis_conn.exists(f"job:{job_id}"):
        return {"error": "Job not found"}
    
    job_data = redis_conn.hgetall(f"job:{job_id}")
    status = job_data.get("status")
    
    if status != "completed":
        return {
            "error": f"Job not completed yet. Current status: {status}",
            "job_id": job_id,
            "status": status
        }
    
    # Return results
    results = json.loads(job_data.get("results", "[]"))
    summary = json.loads(job_data.get("summary", "{}"))
    
    return {
        "job_id": job_id,
        "status": "completed",
        "results": results,
        "summary": summary,
        "total_processed": len(results)
    }


@app.delete("/job/{job_id}")
def delete_job(job_id: str):
    """
    Delete a job and its results.
    """
    if not redis_available:
        return {"error": "Redis not available"}
    
    if redis_conn.exists(f"job:{job_id}"):
        redis_conn.delete(f"job:{job_id}")
        return {"message": f"Job {job_id} deleted successfully"}
    
    return {"error": "Job not found"}
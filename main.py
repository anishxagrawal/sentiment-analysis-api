# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from typing import List
import uuid
import json
import math
from redis import Redis
from rq import Queue

from jobs import process_batch_job

# ==================== REDIS SETUP ====================

try:
    redis_conn = Redis(host="localhost", port=6379, db=0, decode_responses=True)
    task_queue = Queue("default", connection=redis_conn)
    redis_available = True
except Exception as e:
    print("Redis connection failed:", e)
    redis_available = False

# ==================== APP ====================

app = FastAPI()

sentiment_classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

request_count = 0

# ==================== CALIBRATION ====================

def get_calibrated_confidence(raw_score: float, sentiment: str) -> float:
    if raw_score >= 0.99:
        calibrated = 0.75 + (raw_score - 0.99) * 5
    elif raw_score >= 0.95:
        calibrated = 0.65 + (raw_score - 0.95) * 2.5
    elif raw_score >= 0.90:
        calibrated = 0.60 + (raw_score - 0.90)
    elif raw_score >= 0.80:
        calibrated = 0.55 + (raw_score - 0.80) * 0.5
    else:
        calibrated = raw_score * 0.7

    calibrated = min(calibrated, raw_score)

    if sentiment == "NEGATIVE":
        calibrated *= 0.95

    return round(calibrated, 4)

# ==================== MODELS ====================

class TextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: List[str]

class PredictionResult(BaseModel):
    text: str
    sentiment: str
    raw_confidence: float
    calibrated_confidence: float
    confidence_level: str

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

class AsyncBatchResponse(BaseModel):
    job_id: str | None
    status: str
    message: str

# ==================== SINGLE PREDICTION ====================

@app.post("/predict", response_model=PredictionResult)
def predict_sentiment(input: TextInput):
    global request_count
    request_count += 1

    text = input.text.strip()
    if len(text) < 3:
        raise HTTPException(status_code=400, detail="Text must be at least 3 characters")

    prediction = sentiment_classifier(text, top_k=None)
    scores = prediction[0]

    pos = next(s["score"] for s in scores if s["label"] == "POSITIVE")
    neg = next(s["score"] for s in scores if s["label"] == "NEGATIVE")

    label = "POSITIVE" if pos >= neg else "NEGATIVE"
    raw_confidence = max(pos, neg)

    calibrated_confidence = get_calibrated_confidence(raw_confidence, label)

    if calibrated_confidence < 0.75 or abs(pos - neg) < 0.2:
        sentiment = "NEUTRAL"
    else:
        sentiment = label

    if calibrated_confidence >= 0.70:
        confidence_level = "high"
    elif calibrated_confidence >= 0.55:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    return PredictionResult(
        text=text,
        sentiment=sentiment,
        raw_confidence=round(raw_confidence, 4),
        calibrated_confidence=calibrated_confidence,
        confidence_level=confidence_level
    )

# ==================== SYNC BATCH ====================

@app.post("/batch-predict", response_model=BatchResponse)
def predict_batch_sentiment(input: BatchTextInput):
    global request_count

    results = []
    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    total_calibrated = 0

    for text in input.texts:
        cleaned = text.strip()
        if len(cleaned) < 3:
            continue

        prediction = sentiment_classifier(cleaned, top_k=None)

        # Normalize HF output
        if isinstance(prediction[0], dict):
            scores = prediction
        else:
            scores = prediction[0]


        pos = next(s["score"] for s in scores if s["label"] == "POSITIVE")
        neg = next(s["score"] for s in scores if s["label"] == "NEGATIVE")

        label = "POSITIVE" if pos >= neg else "NEGATIVE"
        raw_confidence = max(pos, neg)
        calibrated_confidence = get_calibrated_confidence(raw_confidence, label)

        if calibrated_confidence < 0.75 or abs(pos - neg) < 0.2:
            sentiment = "NEUTRAL"
        else:
            sentiment = label

        if calibrated_confidence >= 0.70:
            confidence_level = "high"
        elif calibrated_confidence >= 0.55:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        results.append(
            BatchPredictionResult(
                text=cleaned,
                sentiment=sentiment,
                raw_confidence=round(raw_confidence, 4),
                calibrated_confidence=calibrated_confidence,
                confidence_level=confidence_level
            )
        )

        sentiment_counts[sentiment] += 1
        total_calibrated += calibrated_confidence
        request_count += 1

    total_processed = len(results)
    avg_calibrated = total_calibrated / total_processed if total_processed else 0

    summary = {
        "total_texts": len(input.texts),
        "successfully_processed": total_processed,
        "positive_count": sentiment_counts["POSITIVE"],
        "negative_count": sentiment_counts["NEGATIVE"],
        "neutral_count": sentiment_counts["NEUTRAL"],
        "average_calibrated_confidence": round(avg_calibrated, 4),
        "most_common_sentiment": max(sentiment_counts, key=sentiment_counts.get)
    }

    return BatchResponse(
        results=results,
        total_processed=total_processed,
        summary=summary
    )

# ==================== ASYNC BATCH ====================

@app.post("/batch-predict-async", response_model=AsyncBatchResponse)
def submit_batch_job_async(input: BatchTextInput):
    if not redis_available:
        return {
            "job_id": None,
            "status": "error",
            "message": "Redis not available"
        }

    job_id = str(uuid.uuid4())

    redis_conn.hset(
        f"job:{job_id}",
        mapping={"status": "queued", "total": len(input.texts), "processed": 0, "percent": 0}
    )

    task_queue.enqueue(
        process_batch_job,
        job_id,
        input.texts,
        redis_conn,           # ✅ FIXED: pass redis_conn
        job_timeout="10m"
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Job submitted ({len(input.texts)} texts)"
    }

# ==================== JOB STATUS ====================

@app.get("/job/{job_id}/status")
def get_job_status(job_id: str):
    if not redis_conn.exists(f"job:{job_id}"):
        return {"error": "Job not found"}

    data = redis_conn.hgetall(f"job:{job_id}")
    return {
        "job_id": job_id,
        "status": data.get("status"),
        "processed": data.get("processed"),
        "total": data.get("total"),
        "percent": data.get("percent"),
    }

@app.get("/job/{job_id}/results")
def get_job_results(job_id: str):
    if not redis_conn.exists(f"job:{job_id}"):
        return {"error": "Job not found"}

    data = redis_conn.hgetall(f"job:{job_id}")
    if data.get("status") != "completed":
        return {"status": data.get("status")}

    return {
        "results": json.loads(data["results"]),
        "summary": json.loads(data["summary"])
    }

# ==================== MISC ====================

@app.get("/stats")
def get_stats():
    return {"total_predictions": request_count}

@app.get("/")
def home():
    return {
        "message": "Welcome to Sentiment API",
        "docs": "/docs",
        "async_enabled": redis_available
    }

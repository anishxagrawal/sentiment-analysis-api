from transformers import pipeline
from typing import List, Dict
import time

# Load model (will be loaded once per worker)
sentiment_classifier = pipeline("sentiment-analysis", 
                                model="distilbert-base-uncased-finetuned-sst-2-english")

def process_batch_job(job_id: str, texts: List[str], redis_conn):
    """
    Background job that processes batch sentiment analysis.
    Updates progress in Redis as it goes.
    """
    
    total = len(texts)
    results = []
    sentiments_count = {"POSITIVE": 0, "NEGATIVE": 0}
    total_confidence = 0
    
    # Update job status to processing
    redis_conn.hset(f"job:{job_id}", "status", "processing")
    redis_conn.hset(f"job:{job_id}", "total", total)
    redis_conn.hset(f"job:{job_id}", "processed", 0)
    
    for idx, text in enumerate(texts):
        # Clean and validate
        cleaned_text = text.strip()
        
        if len(cleaned_text) == 0 or len(cleaned_text) < 3:
            continue
        
        # Predict
        prediction = sentiment_classifier(cleaned_text)
        label = prediction[0]["label"]
        confidence = prediction[0]["score"]
        
        # Store result
        result = {
            "text": cleaned_text,
            "sentiment": label,
            "confidence": confidence
        }
        results.append(result)
        
        # Update statistics
        sentiments_count[label] = sentiments_count.get(label, 0) + 1
        total_confidence += confidence
        
        # Update progress in Redis (every text)
        processed = idx + 1
        redis_conn.hset(f"job:{job_id}", "processed", processed)
        redis_conn.hset(f"job:{job_id}", "percent", int((processed / total) * 100))
        
        # Small delay to simulate realistic processing (remove in production)
        time.sleep(0.1)
    
    # Calculate summary
    total_processed = len(results)
    avg_confidence = total_confidence / total_processed if total_processed > 0 else 0
    
    summary = {
        "total_texts": total,
        "successfully_processed": total_processed,
        "positive_count": sentiments_count.get("POSITIVE", 0),
        "negative_count": sentiments_count.get("NEGATIVE", 0),
        "average_confidence": round(avg_confidence, 4),
        "most_common_sentiment": max(sentiments_count, key=sentiments_count.get) if sentiments_count else "NONE"
    }
    
    # Store results and summary in Redis
    import json
    redis_conn.hset(f"job:{job_id}", "results", json.dumps(results))
    redis_conn.hset(f"job:{job_id}", "summary", json.dumps(summary))
    redis_conn.hset(f"job:{job_id}", "status", "completed")
    
    # Set expiration (results deleted after 1 hour)
    redis_conn.expire(f"job:{job_id}", 3600)
    
    return {"job_id": job_id, "total_processed": total_processed}
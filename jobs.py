from redis import Redis
from transformers import pipeline
import json

def process_batch_job(job_id: str, texts: list[str], redis_conn):
    """
    Background job that processes batch sentiment analysis.
    Updates progress in Redis as it goes.
    """
    
    total = len(texts)
    results = []
    sentiments_count = {"POSITIVE": 0, "NEGATIVE": 0}
    total_raw_confidence = 0
    total_calibrated_confidence = 0
    
    # Update job status to processing
    redis_conn.hset(f"job:{job_id}", "status", "processing")
    redis_conn.hset(f"job:{job_id}", "total", total)
    redis_conn.hset(f"job:{job_id}", "processed", 0)
    
    for idx, text in enumerate(texts):
        # Clean and validate
        cleaned_text = text.strip()
        
        if len(cleaned_text) == 0 or len(cleaned_text) < 3:
            continue
        
        # Predict with all scores
        prediction = sentiment_classifier(cleaned_text, return_all_scores=True)
        scores = prediction[0]
        
        # Get predicted label and raw confidence
        predicted = max(scores, key=lambda x: x['score'])
        label = predicted['label']
        raw_confidence = predicted['score']
        
        # Import the calibration function
        from main import get_calibrated_confidence
        
        # Calculate calibrated confidence
        calibrated_confidence = get_calibrated_confidence(raw_confidence, label)
        
        # Determine confidence level
        if calibrated_confidence >= 0.70:
            confidence_level = "high"
        elif calibrated_confidence >= 0.55:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # Store result
        result = {
            "text": cleaned_text,
            "sentiment": label,
            "raw_confidence": round(raw_confidence, 4),
            "calibrated_confidence": calibrated_confidence,
            "confidence_level": confidence_level
        }
        results.append(result)
        
        # Update statistics
        sentiments_count[label] = sentiments_count.get(label, 0) + 1
        total_raw_confidence += raw_confidence
        total_calibrated_confidence += calibrated_confidence
        
        # Update progress in Redis
        processed = idx + 1
        redis_conn.hset(f"job:{job_id}", "processed", processed)
        redis_conn.hset(f"job:{job_id}", "percent", int((processed / total) * 100))
        
        time.sleep(0.1)  # Remove in production
    
    # Calculate summary
    total_processed = len(results)
    avg_raw_confidence = total_raw_confidence / total_processed if total_processed > 0 else 0
    avg_calibrated_confidence = total_calibrated_confidence / total_processed if total_processed > 0 else 0
    
    # Count confidence levels
    confidence_distribution = {
        "high": sum(1 for r in results if r['confidence_level'] == 'high'),
        "medium": sum(1 for r in results if r['confidence_level'] == 'medium'),
        "low": sum(1 for r in results if r['confidence_level'] == 'low')
    }
    
    summary = {
        "total_texts": total,
        "successfully_processed": total_processed,
        "positive_count": sentiments_count.get("POSITIVE", 0),
        "negative_count": sentiments_count.get("NEGATIVE", 0),
        "average_raw_confidence": round(avg_raw_confidence, 4),
        "average_calibrated_confidence": round(avg_calibrated_confidence, 4),
        "confidence_distribution": confidence_distribution,
        "most_common_sentiment": max(sentiments_count, key=sentiments_count.get) if sentiments_count else "NONE"
    }
    
    # Store results and summary
    import json
    redis_conn.hset(f"job:{job_id}", "results", json.dumps(results))
    redis_conn.hset(f"job:{job_id}", "summary", json.dumps(summary))
    redis_conn.hset(f"job:{job_id}", "status", "completed")
    
    # Set expiration
    redis_conn.expire(f"job:{job_id}", 3600)
    
    return {"job_id": job_id, "total_processed": total_processed}
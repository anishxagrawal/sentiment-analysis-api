from redis import Redis
from transformers import pipeline
import time
import json

# Load model inside worker process
sentiment_classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    top_k=None
)

def process_batch_job(job_id: str, texts: list[str]):
    """
    Background job for async batch sentiment analysis.
    """

    # ✅ Create Redis connection INSIDE the job
    redis_conn = Redis(host="localhost", port=6379, db=0, decode_responses=True)

    from main import get_calibrated_confidence

    total = len(texts)
    results = []
    sentiments_count = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

    redis_conn.hset(
        f"job:{job_id}",
        mapping={
            "status": "processing",
            "total": total,
            "processed": 0,
            "percent": 0,
        },
    )

    for idx, text in enumerate(texts):
        cleaned_text = text.strip()
        if len(cleaned_text) < 3:
            continue

        prediction = sentiment_classifier(cleaned_text)
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

        results.append(
            {
                "text": cleaned_text,
                "sentiment": sentiment,
                "raw_confidence": round(raw_confidence, 4),
                "calibrated_confidence": calibrated_confidence,
            }
        )

        sentiments_count[sentiment] += 1

        processed = idx + 1
        redis_conn.hset(
            f"job:{job_id}",
            mapping={
                "processed": processed,
                "percent": int((processed / total) * 100),
            },
        )

        time.sleep(0.05)  # remove in production

    summary = {
        "total_texts": total,
        "successfully_processed": len(results),
        "positive_count": sentiments_count["POSITIVE"],
        "negative_count": sentiments_count["NEGATIVE"],
        "neutral_count": sentiments_count["NEUTRAL"],
        "most_common_sentiment": max(sentiments_count, key=sentiments_count.get),
    }

    redis_conn.hset(
        f"job:{job_id}",
        mapping={
            "status": "completed",
            "results": json.dumps(results),
            "summary": json.dumps(summary),
        },
    )

    redis_conn.expire(f"job:{job_id}", 3600)

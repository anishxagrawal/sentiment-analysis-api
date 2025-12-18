from redis import Redis
from transformers import pipeline
import json

def process_batch_job(job_id, texts):
    # Create Redis connection INSIDE worker
    redis_conn = Redis(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=True
    )

    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    redis_conn.hset(f"job:{job_id}", "status", "processing")

    results = []
    sentiments = {"POSITIVE": 0, "NEGATIVE": 0}
    total = len(texts)

    for i, text in enumerate(texts, start=1):
        text = text.strip()
        if len(text) < 3:
            continue

        pred = classifier(text)[0]

        results.append({
            "text": text,
            "sentiment": pred["label"],
            "confidence": pred["score"]
        })

        sentiments[pred["label"]] += 1

        redis_conn.hset(
            f"job:{job_id}",
            mapping={
                "processed": i,
                "percent": int((i / total) * 100)
            }
        )

    redis_conn.hset(
        f"job:{job_id}",
        mapping={
            "status": "completed",
            "results": json.dumps(results),
            "summary": json.dumps(sentiments)
        }
    )

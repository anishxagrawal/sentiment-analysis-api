---
title: "Sentiment Analysis API"
emoji: "🎭"
colorFrom: "blue"
colorTo: "purple"
sdk: "docker"
app_file: "Dockerfile"
pinned: false
---

# 🎭 Sentiment Analysis API

A production-ready REST API for real-time sentiment analysis using DistilBERT.

## 🚀 Live Demo

**API Endpoint:** `https://anishxagrawal-sentiment-analysis-api.hf.space`

**Try it:** [Interactive Docs](https://anishxagrawal-sentiment-analysis-api.hf.space/docs)

## ✨ Features

- 🤖 **ML-Powered**: Uses DistilBERT transformer model
- ⚡ **Fast API**: Built with FastAPI for high performance
- 📊 **Request Tracking**: Monitor usage with `/stats` endpoint
- ✅ **Input Validation**: Automatic error handling
- 📚 **Auto Documentation**: Swagger UI at `/docs`

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **ML Library**: Hugging Face Transformers
- **Model**: DistilBERT (fine-tuned for sentiment)
- **Deployment**: Hugging Face Spaces (Docker)
- **Language**: Python 3.11

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| POST | `/predict` | Analyze sentiment of text |
| GET | `/stats` | View request statistics |
| GET | `/docs` | Interactive API documentation |

## 💡 Example Usage

**Request:**
```json
POST /predict
{
  "text": "I love this product!"
}
```

**Response:**
```json
{
  "text": "I love this product!",
  "sentiment": "POSITIVE",
  "confidence": 0.9998
}
```

## 👨‍💻 Author

**Anish Agrawal** - CS Student | Building AI/ML Portfolio

---

Built as part of internship preparation project series.

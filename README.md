# PrioritizeAI

AI-powered customer complaint prioritization system that predicts **Low, Medium, and High severity** using complaint text, sentiment, and network KPI data.

### Key Features
- **TF-IDF + Transformer Sentiment Analysis** for complaint text
- **XGBoost** for multiclass severity classification
- **Network KPI integration** for better prioritization
- **FastAPI** endpoint for real-time predictions

### Tech Stack
**Python | Pandas | Scikit-learn | XGBoost | Transformers | FastAPI**

### Workflow
```text
Complaint → Sentiment + TF-IDF + KPI → XGBoost → Severity → Priority Score
```

### API
Run:
```bash
python -m uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs` to test predictions.

> Built as a learning/demo project using a small sample dataset.
# PrioritizeAI

### AI-Powered Customer Complaint Prioritization

PrioritizeAI is a machine learning system that analyzes telecom customer complaints and prioritizes them based on customer sentiment, complaint characteristics, and network performance KPIs.

## Features

- Sentiment analysis using a transformer-based NLP model
- TF-IDF based text feature extraction
- Integration of network KPI data
- XGBoost-based complaint severity classification
- Automated complaint priority scoring
- FastAPI endpoint for real-time predictions

## Workflow

Customer Complaints + Network KPIs  
↓  
Sentiment Analysis  
↓  
TF-IDF & Feature Engineering  
↓  
XGBoost Severity Prediction  
↓  
Priority Scoring  
↓  
Prioritized Complaints

## Tech Stack

- Python
- Pandas & NumPy
- Scikit-learn
- TF-IDF
- Transformers
- XGBoost
- FastAPI
- Uvicorn

## Project Structure

```text
PrioritizeAI/
├── data/
│   ├── customer_complaints.csv
│   ├── kpi_data.csv
│   ├── new_complaints.csv
│   └── prioritized_complaints.csv
├── models/
│   ├── classifier.pkl
│   └── tfidf_vectorizer.pkl
├── api.py
├── classifier.py
├── data_loader.py
├── feature_engineering.py
├── main.py
├── prioritization.py
├── sentiment_analysis.py
├── requirements.txt
└── README.md

git clone https://github.com/rithikatnit2005/PrioritizeAI.git
cd PrioritizeAI
pip install -r requirements.txt

python main.py

data/prioritized_complaints.csv

uvicorn api:app --reload

http://127.0.0.1:8000/docs

{
  "complaint_id": 107,
  "area_id": "A1",
  "complaint_text": "My internet connection keeps failing and the service is terrible"
}


### Step 2 — Save

Press:

```text
Ctrl + S
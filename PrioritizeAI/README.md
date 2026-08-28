# PrioritizeAI

### AI-Powered Customer Complaint Prioritization

PrioritizeAI is an AI-driven system that analyzes customer complaints and prioritizes them based on **sentiment, complaint features, and network KPI data**.

The project combines **Natural Language Processing (NLP)** and **Machine Learning** to classify complaint severity and help identify issues that require immediate attention.

## How It Works

```text
Customer Complaints + Network KPI Data
                │
                ▼
        Sentiment Analysis
                │
                ▼
        Feature Engineering
                │
                ▼
         XGBoost Classifier
                │
                ▼
      Severity Prediction
                │
                ▼
      Complaint Prioritization
```

## Features

* Sentiment analysis using a Transformer-based NLP model
* TF-IDF feature extraction from complaint text
* Integration of network KPI data
* Complaint severity prediction using XGBoost
* Automated complaint prioritization
* FastAPI support for real-time predictions

## Tech Stack

* **Python**
* **Pandas & NumPy**
* **Hugging Face Transformers**
* **Scikit-learn**
* **XGBoost**
* **FastAPI & Uvicorn**

## Project Structure

```text
PrioritizeAI/
│
├── data/
│   ├── customer_complaints.csv
│   └── kpi_data.csv
│
├── data_loader.py
├── sentiment_analysis.py
├── feature_engineering.py
├── classifier.py
├── prioritization.py
├── api.py
├── main.py
└── README.md
```

## Installation

```bash
git clone https://github.com/rithikatnit2005/PrioritizeAI.git
cd PrioritizeAI
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost transformers torch fastapi uvicorn
```

Run the project:

```bash
python main.py
```

## API

Start the API using:

```bash
uvicorn api:app --reload
```

## Future Improvements

* Train on larger real-world complaint datasets
* Add an interactive dashboard
* Implement a database for complaint storage
* Deploy the API to the cloud

---

**PrioritizeAI** demonstrates how NLP and Machine Learning can be used to make customer complaint management more intelligent and data-driven.


#

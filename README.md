# PrioritizeAI – Customer Complaint Prioritization

PrioritizeAI is a simple AI-powered customer complaint prioritization system.

It uses complaint text, sentiment analysis, TF-IDF features, and network KPI data to classify complaints into Low, Medium, and High severity levels.

## Features

- Customer complaint classification using XGBoost
- Transformer-based sentiment analysis
- TF-IDF text feature extraction
- Network KPI integration
- Priority score generation
- FastAPI endpoint for real-time predictions

## Project Workflow

```text
Customer Complaint
        ↓
Sentiment Analysis
        ↓
TF-IDF Feature Extraction
        ↓
Network KPI Data
        ↓
Feature Combination
        ↓
XGBoost Classifier
        ↓
Severity Prediction
        ↓
Priority Score
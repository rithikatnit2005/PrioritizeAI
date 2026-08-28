import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


def create_features(complaints, kpi_data, fit=True):
    complaints = complaints.copy()

    text = complaints["complaint_text"].fillna("").astype(str)

    if fit:
        vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        text_features = vectorizer.fit_transform(text)

        os.makedirs("models", exist_ok=True)
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(vectorizer, f)
    else:
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        text_features = vectorizer.transform(text)

    length = text.str.len().to_numpy().reshape(-1, 1)

    sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
    sentiment = (
        complaints["sentiment"]
        .astype(str)
        .str.lower()
        .map(sentiment_map)
        .fillna(0)
        .to_numpy()
        .reshape(-1, 1)
    )

    data = complaints.merge(kpi_data, on="area_id", how="left")

    kpi_cols = [
        "access_issue",
        "drop_call_rate",
        "voice_quality_score",
        "data_throughput"
    ]

    kpis = data[kpi_cols].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0).to_numpy()

    numerical = csr_matrix(
        np.hstack([length, sentiment, kpis])
    )

    X = hstack([text_features, numerical]).tocsr()

    if "severity" in complaints:
        severity = complaints["severity"]

        if severity.dtype == "object":
            mapping = {"low": 0, "medium": 1, "high": 2}
            y = severity.astype(str).str.lower().map(mapping)
        else:
            y = severity

        y = y.fillna(0).astype(int).to_numpy()
    else:
        y = None

    return X, y
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentiment_analysis import analyze_sentiment


def create_features(complaints, kpi_data, fit=True):

    # Convert complaint text into TF-IDF features
    vectorizer = TfidfVectorizer(max_features=20)

    if fit:
        text_features = vectorizer.fit_transform(
            complaints["complaint_text"]
        )

        with open("models/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

    else:
        with open("models/tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        text_features = vectorizer.transform(
            complaints["complaint_text"]
        )

    # Get sentiment score
    sentiment_scores = complaints["complaint_text"].apply(
        lambda x: analyze_sentiment(x)[1]
    )

    # Add sentiment as a numerical feature
    sentiment_features = sentiment_scores.values.reshape(-1, 1)

    # Add network KPI information
    data = complaints.merge(kpi_data, on="area_id", how="left")

    kpi_features = data[
        ["drop_call_rate", "data_throughput"]
    ].values

    # Combine all features
    from scipy.sparse import hstack

    X = hstack([
        text_features,
        sentiment_features,
        kpi_features
    ])

    # Severity is the target variable
    if "severity" in complaints.columns:
        mapping = {"Low": 0, "Medium": 1, "High": 2}
        y = complaints["severity"].map(mapping)

        return X, y

    return X, None
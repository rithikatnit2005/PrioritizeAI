import pickle
import pandas as pd

from sentiment_analysis import analyze_sentiment
from feature_engineering import create_features


def prioritize_issues(
    model_path,
    new_complaints,
    kpi_data
):
    """
    Predict severity for new complaints
    and prioritize them.
    """

    # -------------------------------------------------
    # Load trained model
    # -------------------------------------------------

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    # -------------------------------------------------
    # Analyze sentiment
    # -------------------------------------------------

    new_complaints = analyze_sentiment(
        new_complaints
    )

    # -------------------------------------------------
    # Create features using the SAME
    # TF-IDF vectorizer used during training
    # -------------------------------------------------

    X_new, _ = create_features(
        new_complaints,
        kpi_data,
        fit=False
    )

    # -------------------------------------------------
    # Predict severity
    # -------------------------------------------------

    predictions = clf.predict(X_new)

    severity_map = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    new_complaints["predicted_severity"] = [
        severity_map.get(
            int(prediction),
            "Unknown"
        )
        for prediction in predictions
    ]

    # -------------------------------------------------
    # Priority score
    # -------------------------------------------------

    severity_score = {
        "Low": 1,
        "Medium": 2,
        "High": 3
    }

    new_complaints["priority_score"] = (
        new_complaints["predicted_severity"]
        .map(severity_score)
    )

    # -------------------------------------------------
    # Sort by priority
    # -------------------------------------------------

    return new_complaints.sort_values(
        by="priority_score",
        ascending=False
    )
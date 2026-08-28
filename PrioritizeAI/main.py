import pandas as pd

from data_loader import load_data
from sentiment_analysis import analyze_sentiment
from feature_engineering import create_features
from classifier import classify_complaints
from prioritization import prioritize_issues


if __name__ == "__main__":

    # -------------------------------------------------
    # Load training data
    # -------------------------------------------------

    complaints, kpi_data = load_data(
        "data/customer_complaints.csv",
        "data/kpi_data.csv"
    )

    # -------------------------------------------------
    # Analyze sentiment
    # -------------------------------------------------

    complaints = analyze_sentiment(
        complaints
    )

    # -------------------------------------------------
    # Create training features
    # -------------------------------------------------

    X, y = create_features(
        complaints,
        kpi_data,
        fit=True
    )

    # -------------------------------------------------
    # Train and save classifier
    # -------------------------------------------------

    clf = classify_complaints(
        X,
        y,
        model_path="models/classifier.pkl"
    )

    # -------------------------------------------------
    # Load new complaints
    # -------------------------------------------------

    new_complaints = pd.read_csv(
        "data/new_complaints.csv"
    )

    # -------------------------------------------------
    # Predict and prioritize
    # -------------------------------------------------

    prioritized_complaints = prioritize_issues(
        "models/classifier.pkl",
        new_complaints,
        kpi_data
    )

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------

    prioritized_complaints.to_csv(
        "data/prioritized_complaints.csv",
        index=False
    )

    # -------------------------------------------------
    # Display results
    # -------------------------------------------------

    print("\n========================================")
    print("      PRIORITIZED COMPLAINTS")
    print("========================================\n")

    print(
        prioritized_complaints[
            [
                "complaint_id",
                "area_id",
                "complaint_text",
                "sentiment",
                "predicted_severity",
                "priority_score"
            ]
        ].to_string(index=False)
    )

    print(
        "\nOutput saved to:"
        "\ndata/prioritized_complaints.csv"
    )
from fastapi import FastAPI
import pandas as pd

from feature_engineering import create_features
from prioritization import prioritize_complaint

app = FastAPI()


@app.post("/predict/")
def predict_complaint(data: dict):

    # Convert incoming complaint to DataFrame
    complaint = pd.DataFrame([data])

    # Load KPI data
    kpi_data = pd.read_csv("data/kpi_data.csv")

    # Create features
    X, _ = create_features(
        complaint,
        kpi_data,
        fit=False
    )

    # Predict severity and priority
    result = prioritize_complaint(X)

    return {
        "complaint_id": data["complaint_id"],
        "complaint_text": data["complaint_text"],
        "severity": result["severity"],
        "priority_score": result["priority_score"]
    }

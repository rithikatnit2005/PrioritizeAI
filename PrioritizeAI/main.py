import pandas as pd

from data_loader import load_data
from feature_engineering import create_features
from classifier import train_model
from prioritization import prioritize_complaint


# 1. Load training data
complaints, kpi_data = load_data()

# 2. Create training features
X, y = create_features(complaints, kpi_data)

# 3. Train the model
model = train_model(X, y)

# 4. Create a new complaint
new_complaint = pd.DataFrame([{
    "complaint_id": 116,
    "area_id": "A1",
    "complaint_text": "My internet connection keeps failing"
}])

# 5. Create features for the new complaint
X_new, _ = create_features(
    new_complaint,
    kpi_data,
    fit=False
)

# 6. Predict severity and priority
result = prioritize_complaint(X_new)

print("\nNEW COMPLAINT")
print("----------------")
print("Complaint:", new_complaint["complaint_text"].iloc[0])
print("Severity:", result["severity"])
print("Priority Score:", result["priority_score"])
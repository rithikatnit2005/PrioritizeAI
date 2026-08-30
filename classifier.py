import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train_model(X, y):

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create XGBoost classifier
    model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="mlogloss"
    )

    # Train the model
    model.fit(X_train, y_train)

    # Test the model
    predictions = model.predict(X_test)

    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    # Save the trained model
    with open("models/classifier.pkl", "wb") as f:
        pickle.dump(model, f)

    return model
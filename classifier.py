import os
import pickle
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def classify_complaints(
    X,
    y,
    model_path="models/classifier.pkl"
):
    """
    Train an XGBoost classifier to predict complaint severity.
    """

    if y is None:
        raise ValueError(
            "Target variable 'severity' is missing."
        )

    # -------------------------------------------------
    # Train-test split
    # -------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    # -------------------------------------------------
    # XGBoost classifier
    # -------------------------------------------------

    clf = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42
    )

    # -------------------------------------------------
    # Train
    # -------------------------------------------------

    clf.fit(
        X_train,
        y_train
    )

    # -------------------------------------------------
    # Evaluate
    # -------------------------------------------------

    y_pred = clf.predict(X_test)

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # -------------------------------------------------
    # Save model
    # -------------------------------------------------

    os.makedirs(
        os.path.dirname(model_path),
        exist_ok=True
    )

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    print(
        f"Model saved to: {model_path}"
    )

    return clf
import pickle


def prioritize_complaint(features):
    with open("models/classifier.pkl", "rb") as f:
        model = pickle.load(f)

    severity = int(model.predict(features)[0])

    severity_names = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    priority_scores = {
        0: 1,
        1: 2,
        2: 3
    }

    return {
        "severity": severity_names[severity],
        "priority_score": priority_scores[severity]
    }
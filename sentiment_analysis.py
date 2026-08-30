from transformers import pipeline

# Load a pre-trained sentiment model
sentiment_model = pipeline("sentiment-analysis")


def analyze_sentiment(text):
    result = sentiment_model(text)[0]

    return result["label"], result["score"]
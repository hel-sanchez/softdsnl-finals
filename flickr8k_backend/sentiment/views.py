"""
Django view for sentiment prediction.
"""
import os
from rest_framework.decorators import api_view
from rest_framework.response import Response
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to tokenizer and model (relative to this script)
TOKENIZER_FILE = os.path.join(BASE_DIR, "data", "sentiment_tokenizer.pkl")
SENTIMENT_MODEL_FILE = os.path.join(BASE_DIR, "data", "sentiment_model.h5")


# lazy load
_tokenizer = None
_model = None
MAX_LEN = 50

def load_tools():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = pickle.load(open(TOKENIZER_FILE,"rb"))
    if _model is None:
        import tensorflow as tf
        _model = tf.keras.models.load_model(SENTIMENT_MODEL_FILE)
    return _tokenizer, _model

@api_view(["POST"])
def predict_sentiment(request):
    data = request.data
    text = data.get("text", "")
    if not text:
        return Response({"error":"No text provided"}, status=400)
    tokenizer, model = load_tools()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred = model.predict(padded)[0][0]
    sentiment = "positive" if pred >= 0.5 else "negative"
    return Response({"text": text, "sentiment": sentiment, "confidence": float(pred)})
from rest_framework.decorators import api_view
from rest_framework.response import Response
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import pickle
from PIL import Image

# Local utilities
from image_caption.utils.preprocess import preprocess_image_file
from image_caption.utils.caption_generator import generate_caption
from image_caption.utils.caption_generator import load_caption_tools  # to ensure lazy loading
from sentiment.views import load_tools as load_sentiment_tools  # you can rename this if needed

# Instantiate feature extractor only once
_extractor = None
def get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    return _extractor


@api_view(["POST"])
def predict_image_sentiment(request):
    if "file" not in request.FILES:
        return Response({"error": "No file uploaded (use key 'file')"}, status=400)

    f = request.FILES["file"]

    # 1. Preprocess image and extract features
    arr = preprocess_image_file(f, target_size=(299, 299))  # shape: (1, 299, 299, 3)
    arr_pp = preprocess_input(arr * 127.5 + 127.5)  # revert normalization
    extractor = get_extractor()
    features = extractor.predict(arr_pp)  # shape: (1, 2048)

    # 2. Generate caption
    caption = generate_caption(features)

    # 3. Predict sentiment of the caption
    tokenizer, model = load_sentiment_tools()
    MAX_LEN = 50
    seq = tokenizer.texts_to_sequences([caption])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred = model.predict(padded)[0][0]
    sentiment = "positive" if pred >= 0.5 else "negative"

    # 4. Return response
    return Response({
        "caption": caption,
        "sentiment": sentiment,
        "confidence": float(pred)
    })

"""
Django view for image captioning endpoint.
"""

from rest_framework.decorators import api_view
from rest_framework.response import Response
from PIL import Image
import numpy as np
import os

# helper modules (from utils)
from .utils.preprocess import preprocess_image_file
from .utils.caption_generator import generate_caption

# Example: we assume we extract features with a fixed extractor at inference
# For simplicity this example uses a precomputed dummy image feature extractor - replace
# with a proper feature extractor (InceptionV3) for production.
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# instantiate extractor once
_extractor = None
def get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    return _extractor

@api_view(["POST"])
def predict_caption(request):
    if "file" not in request.FILES:
        return Response({"error":"No file uploaded (use key 'file')"}, status=400)
    f = request.FILES["file"]
    arr = preprocess_image_file(f, target_size=(299,299))  # returns batch of 1
    # apply inception preprocess & extract features
    arr_pp = preprocess_input(arr*127.5 + 127.5)  # reverse earlier simple scaling for demo
    extractor = get_extractor()
    features = extractor.predict(arr_pp)  # shape (1, 2048)
    caption = generate_caption(features)
    return Response({"caption": caption})
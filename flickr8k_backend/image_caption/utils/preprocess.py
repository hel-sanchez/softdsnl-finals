"""
Utilities to preprocess uploaded image files for inference.
"""

from PIL import Image
import numpy as np

def preprocess_image_file(fp, target_size=(299,299)):
    # fp: a file-like object (Django InMemoryUploadedFile or path)
    img = Image.open(fp).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32")
    # InceptionV3 expects image range preprocessed via keras.applications.inception_v3.preprocess_input
    # but for simplicity here we scale to -1..1
    arr = (arr / 127.5) - 1.0
    arr = np.expand_dims(arr, axis=0)
    return arr
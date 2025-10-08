# 🧠 SOFTDSNL Final Exam — Image Captioning + Sentiment Analysis (Flickr8k)

## 🎯 Objective
In this final exam, you’ll **combine image processing and natural language processing** in a single end-to-end application.  
You will use the **Flickr8k dataset**, which contains 8,000 images with **5 captions each**, to train a **caption generation model** and then perform **sentiment analysis** on the generated captions.

---

## 🧩 Learning Goals
- Integrate **computer vision** and **natural language processing** in one project.
- Build and train a **caption generation model** using CNN + LSTM.
- Apply **sentiment analysis** to the generated captions.
- Deploy your trained models in a **Django REST API**.
- Test model endpoints using **Postman**.

---

## 📦 Dataset
- **Dataset:** [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- It contains:
  - ~8,000 images
  - Each image has 5 human-written captions
- You will:
  1. Preprocess images (resize, normalize)
  2. Tokenize and encode text captions
  3. Train a CNN + LSTM model to predict captions from images
  4. Perform sentiment analysis (positive/neutral/negative) on the generated captions

---

## Requirements

```
tensorflow
keras
numpy
pillow
django
nltk
scikit-learn
```

---

## 🏗️ Project Structure
```
flickr8k_project/
│
├── flickr8k_backend/              # Django project
│   ├── manage.py
│   ├── flickr8k_backend/          # Django settings folder
│   │   ├── settings.py
│   │   ├── urls.py
│   │   ├── asgi.py
│   │   └── wsgi.py
│   │
│   ├── image_caption/             # app for image captioning
│   │   ├── views.py
│   │   ├── models.py
│   │   ├── utils/
│   │   │   ├── preprocess.py
│   │   │   └── caption_generator.py
│   │   └── urls.py
│   │
│   ├── sentiment/                 # app for sentiment analysis
│   │   ├── views.py
│   │   ├── analyzer.py
│   │   ├── models.py
│   │   └── urls.py
│   │
│   └── requirements.txt
│
├── model_training/                # for training scripts
│   ├── train_caption_model.py
│   ├── train_sentiment_model.py
│   └── evaluation.ipynb
│
├── data/                          # dataset and preprocessing artifacts
│   ├── images/
│   └── captions.txt
│
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/flickr8k_final.git
cd flickr8k_final
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 2. Download Dataset
Dataset will be provided via USB flash drive, but you can download it here:
https://www.kaggle.com/datasets/adityajn105/flickr8k?resource=download

---

## Training Files

## File: model_training/train_caption_model.py
```python
"""
Train a simplified image-captioning pipeline:
- Extract image features with InceptionV3 (pretrained)
- Prepare tokenized captions and padded sequences
- Train a small decoder (image features + text input -> next-word)
NOTE: This is a simplified educational example. Full production captioning requires
more careful batching and teacher-forcing loops.
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
from glob import glob

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# ---------- CONFIG ----------
IMAGES_DIR = "../data/images"      # relative to model_training/
CAPTIONS_FILE = "../data/captions.txt"
FEATURES_FILE = "data/image_features.npy"
CAPTION_SEQS_FILE = "data/encoded_captions.npy"
TOKENIZER_FILE = "data/tokenizer.pkl"
CAPTION_MODEL_FILE = "caption_model.h5"

IMG_SHAPE = (299, 299)  # InceptionV3 input
EMBED_DIM = 256
MAX_WORDS = 10000
MAX_LEN = 30
# ----------------------------

# 1) Simple loader for captions.txt (Flickr8k format)
# Assumes each line: "1000268201_693b08cb0e.jpg#0\tA child in a pink dress is climbing up a set of stairs in an entry way ."
def load_captions(fname):
    captions_dict = {}
    with open(fname, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_caps = line.split('\t')
            if len(img_caps) != 2:
                continue
            img_id, caption = img_caps
            img_file = img_id.split('#')[0]
            captions_dict.setdefault(img_file, []).append(caption.lower())
    return captions_dict

# 2) Extract image features with InceptionV3
def extract_image_features(image_paths):
    print("Loading InceptionV3 for feature extraction...")
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    features = {}
    for p in tqdm(image_paths):
        img = tf.keras.preprocessing.image.load_img(p, target_size=IMG_SHAPE)
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = base_model.predict(x)
        fname = os.path.basename(p)
        features[fname] = feat.flatten()
    return features

# 3) Prepare sequences from captions (encoder input / decoder target)
def create_tokenizer(captions_list):
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(captions_list)
    return tokenizer

def encode_captions(captions_dict, tokenizer):
    sequences = []
    image_idxs = []
    for img_file, caps in captions_dict.items():
        for cap in caps:
            seq = tokenizer.texts_to_sequences([cap])[0]
            if len(seq) < 1:
                continue
            # create input-output pairs for each word in the caption
            for i in range(1, len(seq)):
                in_seq = seq[:i]
                out_seq = seq[i]
                in_seq_padded = pad_sequences([in_seq], maxlen=MAX_LEN)[0]
                out_seq_categ = to_categorical([out_seq], num_classes=len(tokenizer.word_index) + 1)[0]
                sequences.append((img_file, in_seq_padded, out_seq_categ))
    return sequences

# 4) Build a simple decoder model that merges image features + text embedding
def build_caption_model(vocab_size):
    # image feature input
    inputs1 = Input(shape=(2048,))
    fe1 = Dense(EMBED_DIM, activation='relu')(inputs1)

    # sequence input
    inputs2 = Input(shape=(MAX_LEN,))
    se1 = Embedding(vocab_size, EMBED_DIM, mask_zero=True)(inputs2)
    se2 = LSTM(256)(se1)

    # combine
    decoder1 = add([fe1, se2])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def main():
    # load captions
    caps = load_captions(CAPTIONS_FILE)
    all_captions = [c for caplist in caps.values() for c in caplist]

    # tokenizer
    tokenizer = create_tokenizer(all_captions)
    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
    print("Vocab size:", vocab_size)

    # save tokenizer
    os.makedirs("data", exist_ok=True)
    pickle.dump(tokenizer, open(TOKENIZER_FILE, "wb"))

    # prepare image list (limit for demo)
    image_paths = glob(os.path.join(IMAGES_DIR, "*.jpg"))
    image_paths = image_paths[:1000]  # reduce for classroom demo

    # extract features
    features = extract_image_features(image_paths)
    np.save(FEATURES_FILE, features)  # NOTE: dict-saved as object array; optional

    # prepare sequences — simplified approach: create N training samples from random captions
    sequences = []
    for img_file, caplist in caps.items():
        for cap in caplist:
            encoded = tokenizer.texts_to_sequences([cap])[0]
            if len(encoded) < 2:
                continue
            # build simple in/out pair: full in_seq -> next token (last)
            in_seq = pad_sequences([encoded[:-1]], maxlen=MAX_LEN)[0]
            out_seq = to_categorical([encoded[-1]], num_classes=vocab_size)[0]
            sequences.append((img_file, in_seq, out_seq))

    # build dataset arrays (filter by features available)
    X1, X2, y = [], [], []
    for img_file, in_seq, out_seq in sequences:
        if img_file in features:
            X1.append(features[img_file])
            X2.append(in_seq)
            y.append(out_seq)
    X1 = np.array(X1)
    X2 = np.array(X2)
    y = np.array(y)
    print("Training samples:", X1.shape)

    # build model
    model = build_caption_model(vocab_size)
    model.summary()

    # train model (small epochs for demo)
    model.fit([X1, X2], y, epochs=5, batch_size=32, validation_split=0.1)

    # save model
    model.save(CAPTION_MODEL_FILE)
    print("Saved caption model:", CAPTION_MODEL_FILE)

if __name__ == "__main__":
    main()
```

---

## File: model_training/train_sentiment_model.py
```python
"""
Train a simple sentiment classifier (binary: positive/negative) using Flickr captions.
This example builds a tiny dataset by labeling captions using a naive rule (for demo).
In a real project use a labeled sentiment dataset or label captions properly.
"""

import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from datasets import load_dataset

# ---------- CONFIG ----------
TOKENIZER_FILE = "data/sentiment_tokenizer.pkl"
SENTIMENT_MODEL_FILE = "sentiment_model.h5"
MAX_WORDS = 5000
MAX_LEN = 50
# --------------------------

def build_dataset_from_captions(captions_filepath, limit=5000):
    import csv
    texts = []
    labels = []

    print("Building sentiment dataset from captions...", captions_filepath)

    positive_keywords = {"happy", "smile", "smiling", "beautiful", "love", "lovely", "cute", "fun"}
    negative_keywords = {"sad", "cry", "crying", "angry", "hate", "bad", "ugly", "broken"}

    # Open the CSV file properly
    with open(captions_filepath, "r", encoding="utf8") as f:
        reader = csv.DictReader(f)  # reads header: image, caption
        for row in reader:
            if len(texts) >= limit:
                break
            caption = row["caption"].strip().lower()
            texts.append(caption)

            # simple rule-based labeling
            if any(w in caption for w in negative_keywords):
                labels.append(0)
            elif any(w in caption for w in positive_keywords):
                labels.append(1)
            else:
                labels.append(1)  # default positive/neutral

    print(f"Loaded {len(texts)} captions for sentiment training.")
    return texts, np.array(labels)

def main():
    texts, labels = build_dataset_from_captions("../data/captions.txt", limit=2000)
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

    # build model
    model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # train
    model.fit(padded, labels, epochs=5, batch_size=32, validation_split=0.1)

    # save tokenizer and model
    os.makedirs("data", exist_ok=True)
    pickle.dump(tokenizer, open(TOKENIZER_FILE, "wb"))
    model.save(SENTIMENT_MODEL_FILE)
    print("Saved sentiment model:", SENTIMENT_MODEL_FILE)

if __name__ == "__main__":
    main()
```

---

## Django Setup

### Create the Django Project (if not yet created)

If you haven't set up the base project folder yet:

```bash
django-admin startproject flickr8k_backend .
```

> The `.` ensures it creates files in the current directory instead of a subfolder.

---

### Create the Django Apps

```bash
python manage.py startapp image_caption
python manage.py startapp sentiment
```

Then open `flickr8k_backend/settings.py` and **add both apps** to `INSTALLED_APPS`:

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'image_caption',
    'sentiment',
]
```

---

## File: flickr8k_backend/image_caption/utils/preprocess.py
```python
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
```

---

## File: flickr8k_backend/image_caption/utils/caption_generator.py
```python
"""
Caption generation helper:
- Load caption model and tokenizer
- Generate a caption from extracted image features (simplified greedy decoding)
"""

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

TOKENIZER_FILE = "data/tokenizer.pkl"
CAPTION_MODEL_FILE = "caption_model.h5"
MAX_LEN = 30

tokenizer = None
caption_model = None

def load_caption_tools():
    global tokenizer, caption_model
    if tokenizer is None:
        tokenizer = pickle.load(open(TOKENIZER_FILE, "rb"))
    if caption_model is None:
        caption_model = load_model(CAPTION_MODEL_FILE)
    return tokenizer, caption_model

def generate_caption(image_feature_vector):
    """
    image_feature_vector: numpy array shape (1, 2048) (features extracted by Inception/ResNet)
    Greedy decode: start token omitted for brevity; token mapping assumed.
    """
    tokenizer, model = load_caption_tools()
    inv_map = {v:k for k,v in tokenizer.word_index.items()}
    # start with empty sequence
    in_text = []
    for i in range(MAX_LEN):
        seq = tokenizer.texts_to_sequences([" ".join(in_text)])[0]
        seq = pad_sequences([seq], maxlen=MAX_LEN)
        yhat = model.predict([image_feature_vector, seq], verbose=0)
        y_index = np.argmax(yhat)
        word = inv_map.get(y_index, None)
        if word is None:
            break
        in_text.append(word)
        if word == 'endseq':
            break
    caption = " ".join(in_text)
    return caption.replace("endseq", "").strip()
```

---

## File: flickr8k_backend/image_caption/views.py
```python
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
```

---

## File: flickr8k_backend/sentiment/views.py
```python
"""
Django view for sentiment prediction.
"""

from rest_framework.decorators import api_view
from rest_framework.response import Response
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

TOKENIZER_FILE = "model_training/data/sentiment_tokenizer.pkl"  # adjust path if needed
SENTIMENT_MODEL_FILE = "model_training/sentiment_model.h5"

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
```

---

## File: flickr8k_backend/image_caption/urls.py
```python
from django.urls import path
from .views import predict_caption

urlpatterns = [
    path("predict_caption/", predict_caption, name="predict_caption"),
]
```

---

## File: flickr8k_backend/sentiment/urls.py
```python
from django.urls import path
from .views import predict_sentiment

urlpatterns = [
    path("predict_sentiment/", predict_sentiment, name="predict_sentiment"),
]
```

---

## File: flickr8k_backend/flickr8k_backend/urls.py
```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/image/', include('image_caption.urls')),
    path('api/text/', include('sentiment.urls')),
]
```

---

## 🌐 Django Integration

### Endpoints
| Method | Endpoint | Description |
|--------|-----------|-------------|
| `POST` | `/api/predict_caption/` | Upload an image → Returns generated caption |
| `POST` | `/api/predict_sentiment/` | Send text (caption) → Returns sentiment |
| `POST` | `/api/predict_image_sentiment/` | Upload image → Returns caption + sentiment |

Example `POST /api/predict_image_sentiment/` Response:
```json
{
  "caption": "A dog jumping over a log in the forest",
  "sentiment": "positive"
}
```

---

## 📸 Postman Requirements
Submit 10 Postman screenshots showing:
- **5 different images** and their generated captions
- **Sentiment results** for each caption
- Include **both image + text endpoints**

---

## 📝 Deliverables

1. ✅ **PDF Report:** `SOFTDSNL_Final_Surnames.pdf`
   - Model training results (loss/accuracy screenshots)
   - Postman screenshots (10 total)
   - Reflection: Describe what worked and what didn’t
   - GitHub repository link

2. ✅ **GitHub Repository:**
   - All source code (Django + training)
   - README.md (this file)
   - Saved model files

---

## 🧮 Grading Criteria (100 pts)

| Criteria | Description | Points |
|----------|--------------|--------|
| Model Training | CNN + LSTM model correctly trained on Flickr8k | 25 |
| Sentiment Analysis | Functional model for classifying caption sentiment | 15 |
| Django Integration | Functional API endpoints with JSON responses | 25 |
| Postman Tests | Clear and complete test screenshots | 15 |
| Report | PDF includes analysis, results, and reflections | 10 |
| Code Quality | Readability, comments, and proper structure | 10 |


---

## 🧑‍🏫 Submission
Submit your:
- **PDF report**
  - **GitHub link**
  - **Postman screenshots**
  - **Enoch Table**
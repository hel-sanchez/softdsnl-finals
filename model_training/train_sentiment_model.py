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
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
import csv

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

def load_captions_csv(fname):
    captions_dict = {}
    with open(fname, "r", encoding="utf8") as f:
        reader = csv.DictReader(f)  # automatically handles header
        for row in reader:
            img_file = row['image'].strip()
            caption = row['caption'].strip().lower()
            if caption:  # skip empty captions
                captions_dict.setdefault(img_file, []).append(caption)
    return captions_dict

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
    caps = load_captions_csv(CAPTIONS_FILE)
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
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # path to .../utils

TOKENIZER_FILE = os.path.join(BASE_DIR, "data", "tokenizer.pkl")  # utils/data/tokenizer.pkl
CAPTION_MODEL_FILE = os.path.join(BASE_DIR, "data", "caption_model.h5")  # adjust path if needed

MAX_LEN = 30

tokenizer = None
caption_model = None



def load_caption_tools():
    global tokenizer, caption_model
    if tokenizer is None:
        with open(TOKENIZER_FILE, "rb") as f:
            tokenizer = pickle.load(f)
    if caption_model is None:
        # ** FIX: The TypeError usually means a custom/Lambda layer's args 
        # are missing. Trying to load without custom_objects first is best, 
        # or include all necessary Keras objects.**
        # If the problem persists, use a custom_objects dict to include 
        # the model's layers (Input, Dense, Embedding, LSTM, etc.)
        
        # In this specific case, if the previous NotEqual fix broke it, 
        # try loading it as is again:
        try:
             caption_model = load_model(CAPTION_MODEL_FILE)
        except ValueError:
             # Fallback to including the missing internal function
             import tensorflow as tf
             custom_objects = {
                 'NotEqual': tf.math.not_equal
             }
             caption_model = load_model(CAPTION_MODEL_FILE, custom_objects=custom_objects)
             
    return tokenizer, caption_model

def generate_caption(image_feature_vector):
    tokenizer, model = load_caption_tools()
    inv_map = {v: k for k, v in tokenizer.word_index.items()}
    in_text = []
    for i in range(MAX_LEN):
        seq = tokenizer.texts_to_sequences([" ".join(in_text)])[0]
        seq = pad_sequences([seq], maxlen=MAX_LEN)
        yhat = model.predict([image_feature_vector, seq], verbose=0)
        y_index = np.argmax(yhat)
        word = inv_map.get(y_index)
        if word is None:
            break
        in_text.append(word)
        if word == 'endseq':
            break
    caption = " ".join(in_text)
    return caption.replace("endseq", "").strip()

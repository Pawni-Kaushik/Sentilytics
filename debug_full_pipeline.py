"""
debug_full_pipeline.py
--------------------------
Run with: python debug_full_pipeline.py
(from your project root folder)

Runs the EXACT same pipeline the app uses (NER masking -> cleaning ->
tokenizing -> sentiment prediction) but prints every intermediate step,
so we can see exactly why a sentence ends up neutral/positive/negative.
"""

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import MODEL_PATH, TOKENIZER_PATH, ENCODER_PATH
from utils.preprocessing import clean_text
from utils.ner import mask_person_entities

print("Loading sentiment model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

print("Loaded. Classes:", list(encoder.classes_))
print()

MAX_LEN = 40  # must match training MAX_LEN in the sentiment notebook


def full_pipeline(text):
    masked_text = mask_person_entities(text)
    cleaned_text = clean_text(masked_text)

    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    print("  original:     ", repr(text))
    print("  after NER:    ", repr(masked_text))
    print("  after cleaning:", repr(cleaned_text))
    print("  token IDs:    ", seq[0])

    oov_id = tokenizer.word_index.get(tokenizer.oov_token) if tokenizer.oov_token else None
    real_tokens = [t for t in seq[0] if t != 0 and t != oov_id]
    print("  real (non-padding, non-OOV) token count:", len(real_tokens))

    probs = model.predict(padded, verbose=0)[0]
    for cls, p in zip(encoder.classes_, probs):
        print(f"    {cls}: {p*100:.1f}%")

    pred = encoder.inverse_transform([np.argmax(probs)])[0]
    print("  FINAL PREDICTION:", pred.upper())
    print()


test_sentences = [
    "Hi i am Happy Singh. I am sad today",
    "Hi i am Happy Singh. I am tired today",
    "Hi i am Happy Singh. I am angry today",
    "sad",
    "angry",
    "tired",
    "hi sad today",
]

for s in test_sentences:
    full_pipeline(s)

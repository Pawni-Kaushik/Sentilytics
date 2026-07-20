"""
debug_ner.py
--------------
Run this directly with: python debug_ner.py
(from your project's root folder, the same folder as app.py)

This tests the NER masking function in isolation, WITHOUT Streamlit's
caching or the try/except that hides errors -- so if something is
broken, you'll see the real error message here instead of it being
silently swallowed.
"""

import pickle
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import NER_MODEL_PATH, NER_WORD2IDX_PATH, NER_IDX2TAG_PATH, NER_CONFIG_PATH

print("Looking for NER files at:")
print(" -", NER_MODEL_PATH, "| exists:", NER_MODEL_PATH.exists())
print(" -", NER_WORD2IDX_PATH, "| exists:", NER_WORD2IDX_PATH.exists())
print(" -", NER_IDX2TAG_PATH, "| exists:", NER_IDX2TAG_PATH.exists())
print(" -", NER_CONFIG_PATH, "| exists:", NER_CONFIG_PATH.exists())
print()

print("Loading NER model...")
model = tf.keras.models.load_model(NER_MODEL_PATH)

with open(NER_WORD2IDX_PATH, "rb") as f:
    word2idx = pickle.load(f)

with open(NER_IDX2TAG_PATH, "rb") as f:
    idx2tag = pickle.load(f)

with open(NER_CONFIG_PATH, "r") as f:
    ner_config = json.load(f)

print("Loaded OK. max_len:", ner_config.get("max_len"), " unk_index:", ner_config.get("unk_index"))
print()


def mask_person_entities(text, placeholder="person"):
    tokens = text.split()
    max_len = ner_config.get("max_len", 75)
    unk_index = ner_config.get("unk_index", 1)

    seq = [word2idx.get(t, unk_index) for t in tokens]
    padded = pad_sequences([seq], maxlen=max_len, padding="post", value=0)

    pred = model.predict(padded, verbose=0)[0]
    pred_tags = [idx2tag[i] for i in np.argmax(pred, axis=-1)][:len(tokens)]

    print("  tokens:", tokens)
    print("  predicted tags:", pred_tags)

    masked_tokens = [
        placeholder if tag in ("B-per", "I-per") else token
        for token, tag in zip(tokens, pred_tags)
    ]
    return " ".join(masked_tokens)


test_sentences = [
    "Happy Singh is sad",
    "hi my name is happy singh . i am sad",
]

for s in test_sentences:
    print("INPUT: ", repr(s))
    result = mask_person_entities(s)
    print("OUTPUT:", repr(result))
    print()

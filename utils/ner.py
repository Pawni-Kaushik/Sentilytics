"""
utils/ner.py
---------------
Loads the separately-trained NER (Named Entity Recognition) model and uses
it to detect and mask PERSON names in raw text, BEFORE sentiment cleaning
runs. This fixes cases like "Happy Singh is sad", where a person's name
happens to also be an English emotion word.

IMPORTANT: masking must run on the ORIGINAL, un-lowercased text.
Capitalization is one of the strongest signals the NER model uses to
detect names -- if this ran after lowercasing, it would barely work.
"""

import pickle
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import NER_MODEL_PATH, NER_WORD2IDX_PATH, NER_IDX2TAG_PATH, NER_CONFIG_PATH


@st.cache_resource
def load_ner_artifacts():
    """Loads and caches the trained NER model, word2idx map, idx2tag map, and config."""
    model = tf.keras.models.load_model(NER_MODEL_PATH)

    with open(NER_WORD2IDX_PATH, "rb") as f:
        word2idx = pickle.load(f)

    with open(NER_IDX2TAG_PATH, "rb") as f:
        idx2tag = pickle.load(f)

    with open(NER_CONFIG_PATH, "r") as f:
        ner_config = json.load(f)

    return model, word2idx, idx2tag, ner_config


def mask_person_entities(text: str) -> str:
    """
    Detects person names in the ORIGINAL (uncapitalized) text using the
    trained NER model, and REMOVES each detected name entirely (rather
    than substituting a placeholder word).

    Why removal instead of a placeholder: an earlier version replaced
    names with the word "person", but "person" appears in only 5 of the
    32,359 training rows -- far too rare for the sentiment model to have
    learned anything reliable about it, and a repeated "person person"
    (from 2-word names) is a pattern the model never saw during training
    at all. That introduced noise instead of a clean neutral signal.
    Simply dropping the name avoids this problem entirely: a sentence
    like "I am Happy Singh" has no real sentiment content once the name
    is removed, and the existing empty-text safeguard in predict_sentiment()
    handles that case honestly (reports neutral, 0% confidence) rather than
    guessing.

    Example: "Happy Singh is sad" -> "is sad"
             "I am Happy Singh" -> "I am"  (then cleaning removes stopwords too)
    """
    try:
        model, word2idx, idx2tag, ner_config = load_ner_artifacts()
    except Exception:
        return text

    tokens = text.split()
    if not tokens:
        return text

    max_len = ner_config.get("max_len", 75)
    unk_index = ner_config.get("unk_index", 1)

    seq = [word2idx.get(t, unk_index) for t in tokens]
    padded = pad_sequences([seq], maxlen=max_len, padding="post", value=0)

    pred = model.predict(padded, verbose=0)[0]
    pred_tags = [idx2tag[i] for i in np.argmax(pred, axis=-1)][:len(tokens)]

    kept_tokens = [
        token for token, tag in zip(tokens, pred_tags)
        if tag not in ("B-per", "I-per")
    ]

    return " ".join(kept_tokens)

"""
utils/loader.py  (v2)
------------------------
Loads the trained v2 model artifacts (LSTM model + Keras Tokenizer +
label encoder), instead of the old TF-IDF vectorizer.

Per project rules: we NEVER retrain or replace these artifacts here.
We only load what was already trained in the v2 notebook.
"""

import pickle
import json
import streamlit as st
import tensorflow as tf

from config import MODEL_PATH, TOKENIZER_PATH, ENCODER_PATH, METRICS_PATH


@st.cache_resource
def load_model_artifacts():
    """Loads and caches the trained model, Tokenizer, and label encoder."""
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)

    return model, tokenizer, encoder


@st.cache_data
def load_precomputed_metrics():
    """Loads the precomputed evaluation metrics/dataset stats JSON."""
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def get_max_len() -> int:
    """Reads the MAX_LEN used at training time (needed to pad new text the same way)."""
    metrics = load_precomputed_metrics()
    return metrics.get("max_len", 40)

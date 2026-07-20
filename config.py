"""
config.py  (v2)
----------
Central configuration: file paths + design tokens (colors, fonts).

Keeping these in one place means every page and utility module pulls
from the same source of truth instead of hardcoding hex values.
"""

from pathlib import Path

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).parent

MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
ASSETS_DIR = BASE_DIR / "assets"
CSS_PATH = ASSETS_DIR / "css" / "style.css"

# --- v2 model artifacts (LSTM + Tokenizer, replacing TF-IDF) ---
MODEL_PATH = MODELS_DIR / "sentiment_model_v2.keras"
TOKENIZER_PATH = MODELS_DIR / "tokenizer_v2.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder_v2.pkl"
METRICS_PATH = MODELS_DIR / "precomputed_metrics_v2.json"

DATASET_PATH = DATASET_DIR / "reddit_sentiment_dataset_v9.csv"
HISTORY_PATH = BASE_DIR / "prediction_history.csv"
USERS_PATH = BASE_DIR / "users.json"

# --- NER model artifacts (person-name detection, used to mask names
#     before sentiment analysis, e.g. "Happy Singh is sad") ---
NER_MODEL_PATH = MODELS_DIR / "ner_model.keras"
NER_WORD2IDX_PATH = MODELS_DIR / "ner_word2idx.pkl"
NER_IDX2TAG_PATH = MODELS_DIR / "ner_idx2tag.pkl"
NER_CONFIG_PATH = MODELS_DIR / "ner_config.json"

# ---------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------
PROJECT_TITLE = "Sentilytics — Sentiment Analyzer"
PROJECT_TAGLINE = "Reddit Sentiment Analysis using NLP and Neural Networks"
DEVELOPERS = ["Pawni Kaushik", "Khushi Sharma"]
PROJECT_YEAR = "2026"

# ---------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------
COLORS = {
    "brand_orange": "#FF4500",
    "positive": "#2FB380",
    "negative": "#7193FF",
    "neutral": "#C9A227",

    "dark_bg": "#0B0E14",
    "dark_surface": "#141821",
    "dark_surface_alt": "#1B2029",
    "dark_text": "#E9EBF0",
    "dark_text_muted": "#8B93A3",
    "dark_border": "#252B37",
    # Glow/shadow intensity tuned per-mode: a 35%-opacity glow that
    # reads as a nice neon accent on a near-black background turns
    # into a muddy smear on white, so light mode uses softer values.
    "dark_glow_alpha": "0.35",

    # Light mode: a slightly deeper off-white than pure white so
    # surfaces actually separate from the page background, plus a
    # darker border/muted-text so elements don't wash out together.
    "light_bg": "#F3F1EC",
    "light_surface": "#FFFFFF",
    "light_surface_alt": "#E9E6DE",
    "light_text": "#1A1A1B",
    "light_text_muted": "#555C68",
    "light_border": "#D8D4CB",
    "light_glow_alpha": "0.16",
}

FONTS = {
    "display": "'Space Grotesk', sans-serif",
    "body": "'Inter', sans-serif",
    "mono": "'JetBrains Mono', monospace",
}

NAV_PAGES = [
    {"label": "Home", "path": "pages/0_Home.py", "icon": "🏠"},
    {"label": "Analyzer", "path": "pages/1_Analyzer.py", "icon": "🚀"},
    {"label": "Live Search", "path": "pages/6_Live_News_Search.py", "icon": "🔎"},
    {"label": "Dashboard", "path": "pages/2_Dashboard.py", "icon": "📊"},
    {"label": "Model", "path": "pages/3_Model.py", "icon": "🧠"},
    {"label": "Performance", "path": "pages/4_Performance.py", "icon": "📈"},
    {"label": "About", "path": "pages/5_About.py", "icon": "ℹ️"},
]

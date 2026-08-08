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
# NOTE: user accounts used to live in a local users.json, but Streamlit
# Community Cloud's disk is ephemeral (wiped on sleep/restart/redeploy),
# so accounts silently disappeared. Auth now uses a hosted Postgres DB
# instead -- see utils/auth.py and DEPLOYMENT.md.

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
    # Raspberry-pink brand accent (was a stock orange-red) and a soft,
    # botanical sentiment palette -- sage for positive, warm clay for
    # negative, honey for neutral -- instead of literal traffic-light
    # red/green, which reads calmer for an emotion-analysis tool.
    "brand_orange": "#D6487A",
    "positive": "#6FAE8A",
    "negative": "#C97C63",
    "neutral": "#D9A66C",

    # Dark mode: a deep forest-charcoal instead of near-black, so it
    # still pairs naturally with the sage/pink accents above.
    "dark_bg": "#1C2620",
    "dark_surface": "#23302A",
    "dark_surface_alt": "#2B3B33",
    "dark_text": "#F2EFE7",
    "dark_text_muted": "#A6B0A3",
    "dark_border": "#384A40",
    "dark_glow_alpha": "0.35",

    # Light mode: warm ivory base (not stark white) so surfaces still
    # separate from the page, with the same sage/pink/clay accents.
    "light_bg": "#FBF8F4",
    "light_surface": "#FFFFFF",
    "light_surface_alt": "#F1EDE3",
    "light_text": "#2E332C",
    "light_text_muted": "#52584D",
    "light_border": "#E4DFD2",
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

# Sentilytics — Sentiment Analyzer

*(Reddit Sentiment Analysis using NLP and Neural Networks)*

Developed by **Pawni Kaushik** & **Khushi Sharma** — Internship Project 2026

A multipage Streamlit application for Reddit-style sentiment analysis, powered by
a custom-trained TF-IDF + Neural Network model (no third-party sentiment APIs —
not HuggingFace, not OpenAI, not any external sentiment service).

## Features

- **Analyzer** — real-time sentiment prediction with confidence gauge, probability
  breakdown, contributing words, batch analysis, and per-session prediction history
  (downloadable as CSV).
- **Dashboard** — dataset statistics: class balance, most common words per class,
  comment length distributions.
- **Model** — plain-language explanation of the NLP + neural network pipeline and
  architecture.
- **Performance** — accuracy, precision/recall/F1, confusion matrix, ROC curves,
  and training/validation curves, all computed from the actual trained model.
- **About** — project description, tech stack, developer credits, known limitations.
- Dark/light mode toggle, glassmorphism UI, Reddit-inspired color theme.

## Project Structure

```
Sentilytics/
├── app.py                     # Home page (multipage entry point)
├── config.py                  # Paths + design tokens
├── requirements.txt
├── news_service.py             # Live News Search: fetches articles via the GNews API
├── assets/
│   └── css/style.css          # Theme stylesheet (dark/light via CSS variables)
├── dataset/
│   └── reddit_sentiment_dataset_v9.csv
├── models/
│   ├── sentiment_model.keras
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   └── precomputed_metrics.json
├── utils/
│   ├── preprocessing.py       # Text cleaning (must match training notebook exactly)
│   ├── loader.py               # Loads model/vectorizer/encoder + cached metrics
│   ├── predictor.py            # predict_sentiment() + explainability helper
│   ├── metrics.py              # Accessors for precomputed evaluation metrics
│   ├── charts.py                # All Plotly figure builders
│   └── helpers.py               # Navbar, theme CSS injection, session state, footer
└── pages/
    ├── 1_Analyzer.py
    ├── 2_Dashboard.py
    ├── 3_Model.py
    ├── 4_Performance.py
    └── 5_About.py
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live News Search

The **Live Search** page fetches real, live news articles for any keyword via
the [GNews API](https://gnews.io) and runs each article through the trained
sentiment model. An earlier version of this feature scraped Reddit directly
with a headless browser; that was dropped because Reddit blocks datacenter/
cloud IPs (which is what most cloud hosts, including Streamlit Community
Cloud, run on), so it failed once deployed even though it worked locally.
GNews is a normal authenticated API call, so it works the same locally and
deployed. Get a free key at gnews.io and set it as `GNEWS_API_KEY` (see
`.env` / `DEPLOYMENT.md`).

## Model Notes

- **Approach:** TF-IDF (bag-of-words) + feedforward neural network (Dense +
  Dropout layers, softmax output).
- **Why not a from-scratch retrain:** per project requirements, this app uses
  the already-trained `sentiment_model.keras` / `tfidf_vectorizer.pkl` /
  `label_encoder.pkl` artifacts as-is.
- **Known limitation:** bag-of-words models don't understand word order,
  negation, or sarcasm. See the About page for documented limitations and
  future improvement ideas (e.g. LSTM/Transformer-based sequence models).

## requirements.txt

See `requirements.txt` for the full dependency list (Streamlit, TensorFlow,
scikit-learn, NLTK, Plotly, Pandas, NumPy, PRAW).

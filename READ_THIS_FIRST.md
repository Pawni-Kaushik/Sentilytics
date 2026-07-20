# READ THIS FIRST — current project status

This is the full project, with ALL code fixes already applied:
- Persistent, filterable prediction history (utils/helpers.py, pages/1_Analyzer.py)
- Context-aware LSTM sentiment model pipeline (config.py, utils/preprocessing.py,
  utils/loader.py, utils/predictor.py)
- Self-trained NER model that detects and removes person names before sentiment
  analysis runs (utils/ner.py)
- Reddit keyword-search function ready for once API credentials are approved
  (reddit_api_template.py)
- Professional homepage with two-model stats + pipeline visual (app.py)

See `FINAL_PROJECT_SPECIFICATION.md` for the full architecture, file structure, and
history of what was built and why.

## IMPORTANT — the app will NOT run yet, on purpose

`config.py` points to model files that are NOT included in this zip (they're large,
locally-trained files, not something regenerated here):

**Sentiment model (from `Reddit_Sentiment_LSTM_Training_v2.ipynb`):**
- models/sentiment_model_v2.keras
- models/tokenizer_v2.pkl
- models/label_encoder_v2.pkl
- models/precomputed_metrics_v2.json

**NER model (from `NER_Training.ipynb`):**
- models/ner_model.keras
- models/ner_word2idx.pkl
- models/ner_idx2tag.pkl
- models/ner_config.json

If you already trained these on your machine before, copy those 8 files from your
previous project folder straight into this new project's `models/` folder -- you do
NOT need to retrain from scratch.

If you're starting fresh: run both notebooks (Kernel -> Restart & Run All), each saves
its own 4 files, copy all 8 into `models/`, then:

```
streamlit run app.py
```

The old v1 files (`sentiment_model.keras`, `tfidf_vectorizer.pkl`, etc.) are still
sitting in `models/` as a harmless backup -- the app code no longer looks for them.

# Sentilytics — Sentiment Analyzer — Final Project Specification

## 1. Project Title
**Sentilytics — Sentiment Analyzer** *(Reddit Sentiment Analysis using NLP and Neural Networks)*

## 2. One-line Purpose
A self-trained NLP pipeline that classifies any piece of text (and eventually live Reddit posts/comments)
as Positive, Negative, or Neutral, using two independently trained neural networks: a sentiment
classifier and a Named Entity Recognition (NER) model, combined into one pipeline — built entirely
in-house on self-collected/labeled data, with no third-party sentiment APIs.

## 3. High-level Architecture

```
Raw text
   |
   v
[1] NER model (BiLSTM) --> detects person names --> REMOVES them
   |
   v
[2] Text Cleaning --> lowercase, strip URLs/punctuation, remove stopwords
    (negation words like "not", "but", "never" are deliberately KEPT)
   |
   v
[3] Tokenizer --> converts words to integer sequences, padded to fixed length
   |
   v
[4] Sentiment model (Embedding + Bidirectional LSTM) --> outputs probabilities
   |
   v
Positive / Negative / Neutral + confidence %
```

Two completely separate trained models are involved:
- **Sentiment model** — classifies a whole (cleaned) sentence as positive/negative/neutral
- **NER model** — classifies each individual word as person/organization/location/etc., used only
  as a pre-processing step to strip out names before sentiment analysis

## 4. Why this architecture (evolution of the project)

The project went through three real iterations, each fixing a specific, demonstrated failure:

1. **v1 — TF-IDF + Dense Neural Network** (bag-of-words). Fast and simple, but had no concept of
   word order — so "It was a good day but I am tired" and sarcasm-driven sentences were
   consistently misclassified.
2. **v2 — Embedding + Bidirectional LSTM** (sequence model). Reads words in order, so it can learn
   patterns like "the word after 'but' usually matters more." Also required a preprocessing fix:
   negation/contrast words (not, no, never, but, however...) were being deleted as generic
   stopwords before the model ever saw them — this was corrected to preserve them.
3. **v3 — NER-augmented pipeline**. Even the LSTM had no way to know a capitalized word like
   "Happy" (in "Happy Singh") was a person's name rather than the emotion word. A second,
   independently trained NER model (BiLSTM tagger) was added to detect and strip person names
   from text before sentiment analysis runs.

Additionally, two rounds of **targeted data augmentation** were used to fix specific, demonstrated
weaknesses in the training data itself (not the architecture):
- **Negation augmentation**: synthetic "this is not X at all" examples, since the base dataset had
  very few explicit negation examples.
- **Emotion-word reinforcement**: investigation of failures uncovered that the base dataset had
  inconsistent labels for common emotion words — e.g. 61% of rows containing "sad" were labeled
  neutral rather than negative. This is a genuine data-quality issue in the source dataset (typical
  of noisy social-media sentiment datasets), fixed by adding clear, unambiguous synthetic examples
  for the affected words so the model has enough clean signal to learn correct polarity.

## 5. Datasets used

| Dataset | Rows | Purpose | Source |
|---|---|---|---|
| `reddit_sentiment_dataset_v9.csv` | 32,359 (+ ~3,900 synthetic augmentation rows) | Sentiment training | Pre-labeled positive/negative/neutral text |
| `ner_dataset.csv` | ~1,000,000 tagged words / ~47,959 sentences (+ 800 synthetic sentences) | NER training | Kaggle: "Annotated Corpus for Named Entity Recognition" (abhinavwalia95) |

## 6. Final trained model results

**Sentiment model (v2, LSTM, with augmentation):**
- Test accuracy: ~88%
- Precision/Recall/F1 per class computed and available in `precomputed_metrics_v2.json`
- Confusion matrix and ROC curves (AUC ~0.97-0.99 across classes) generated and saved

**NER model (BiLSTM tagger, with augmentation):**
- Test accuracy (per-word tagging): ~96.7%
- Correctly identifies PERSON entities in short and multi-word sentence contexts after augmentation

## 7. Documented, honest limitations (for viva/report — do not hide these, they demonstrate understanding)

1. **Sarcasm is not reliably detected.** E.g. "Bro really said trust me and then did the single
   most untrustworthy thing possible" gets misclassified as positive, because sarcasm requires
   understanding tone/intent beyond word patterns — a known hard problem even for large
   state-of-the-art models.
2. **NER accuracy is not 100%.** ~96.7% per-word accuracy means occasional misses are expected,
   especially on unusual sentence structures not well represented in training data.
3. **Data-quality-driven mislabeling can still surface for words/phrases not covered by
   augmentation.** The project's fix targeted the specific words found to be inconsistently
   labeled (sad, angry, happy, excited, etc.) — other words could still carry residual dataset
   noise.
4. **The Reddit Developer API integration is written but not live-tested**, pending Reddit's API
   credential approval. `reddit_api_template.py` contains ready-to-use functions:
   `fetch_subreddit_posts()`, `fetch_subreddit_comments()`, and `fetch_posts_by_keyword()` (the
   last one searches all of Reddit for a keyword/brand and aggregates sentiment across matches).

## 8. Full file structure

```
Sentilytics/
├── app.py                              # Streamlit home page
├── config.py                           # central paths + design tokens
├── reddit_api_template.py              # Reddit API functions (pending credential approval)
├── requirements.txt
├── Reddit_Sentiment_LSTM_Training_v2.ipynb   # sentiment model training notebook (run this to retrain)
├── NER_Training.ipynb                        # NER model training notebook (run this to retrain)
├── debug_ner.py                        # standalone script to test NER masking in isolation
├── debug_full_pipeline.py              # standalone script to test the full NER+sentiment pipeline
├── prediction_history.csv              # auto-generated, saved prediction history (persists across reloads)
│
├── dataset/
│   ├── reddit_sentiment_dataset_v9.csv
│   └── ner/ner_dataset.csv
│
├── models/
│   ├── sentiment_model_v2.keras        # trained sentiment model
│   ├── tokenizer_v2.pkl                # sentiment Tokenizer (word -> integer ID)
│   ├── label_encoder_v2.pkl            # sentiment label encoder (0/1/2 <-> negative/neutral/positive)
│   ├── precomputed_metrics_v2.json     # accuracy, confusion matrix, ROC, training history
│   ├── ner_model.keras                 # trained NER model
│   ├── ner_word2idx.pkl                # NER vocabulary map
│   ├── ner_idx2tag.pkl                 # NER tag map
│   ├── ner_config.json                 # NER max_len, unk_index, test_accuracy
│   └── [older v1 TF-IDF files kept as backup, no longer used by the app]
│
├── pages/
│   ├── 1_Analyzer.py                   # main interactive prediction + batch analysis + history
│   ├── 2_Dashboard.py                  # dataset EDA (class balance, top words, text length)
│   ├── 3_Model.py                      # plain-language explainer of the pipeline/architecture
│   ├── 4_Performance.py                # accuracy/precision/recall/F1/confusion matrix/ROC
│   └── 5_About.py                      # project description, tech stack, known limitations
│
└── utils/
    ├── preprocessing.py                # clean_text() -- lowercase, strip URLs/punctuation, stopwords (negation preserved)
    ├── loader.py                       # loads/caches sentiment model, tokenizer, encoder, metrics
    ├── predictor.py                    # full prediction pipeline: NER masking -> cleaning -> tokenizing -> prediction
    ├── ner.py                          # loads NER model, mask_person_entities() removes detected names
    ├── metrics.py                      # reads precomputed_metrics_v2.json into usable formats
    ├── charts.py                       # all Plotly chart builders
    └── helpers.py                      # navbar, dark mode, persistent prediction history (CSV-backed)
```

## 9. What's still pending / next steps

1. Reddit Developer API credential approval — once received, fill `reddit_api_template.py` with
   real `CLIENT_ID`/`CLIENT_SECRET`, then wire `fetch_subreddit_posts()` /
   `fetch_posts_by_keyword()` into `pages/1_Analyzer.py` or a new page, so users can fetch and
   analyze live Reddit data by subreddit name or keyword.
2. SRS Document, PPT content, Project Report, Viva Q&A — to be generated from this specification.
3. Optional further work: sarcasm handling (would require a much larger, specifically-labeled
   sarcasm dataset — out of scope for this iteration).

## 10. How to reproduce/retrain from scratch

1. Place `dataset/reddit_sentiment_dataset_v9.csv` and `dataset/ner/ner_dataset.csv` as shown above.
2. Run `Reddit_Sentiment_LSTM_Training_v2.ipynb` top to bottom (Kernel → Restart & Run All).
   Produces the 4 `_v2` sentiment files.
3. Run `NER_Training.ipynb` top to bottom. Produces the 4 `ner_*` files.
4. Copy all 8 output files into `models/`.
5. Run `streamlit run app.py`.

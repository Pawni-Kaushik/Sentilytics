"""
utils/predictor.py  (v3 -- with NER masking)
-----------------------------------------------
Wraps the full pipeline: NER masking -> text cleaning -> tokenizing ->
padding -> LSTM sentiment prediction.

New in this version: before any lowercasing/cleaning happens, the raw
text is passed through the trained NER model to detect and mask person
names (e.g. "Happy Singh is sad" -> "person is sad"), so the sentiment
model doesn't confuse a name like "Happy" with the emotion word "happy".
"""

import re

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.preprocessing import clean_text
from utils.loader import get_max_len
from utils.ner import mask_person_entities


def _ensure_vader_resource() -> None:
    """Downloads VADER's lexicon data only if not already present (same pattern as
    ensure_nltk_resources() in preprocessing.py for stopwords)."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")


_ensure_vader_resource()
_vader = SentimentIntensityAnalyzer()


def _clean_percentages(probs: np.ndarray) -> np.ndarray:
    """
    Converts raw softmax probabilities into percentages rounded to 1
    decimal place that always sum to EXACTLY 100.0 -- using the
    "largest remainder" method.

    Why this is needed: rounding each probability independently
    (e.g. 33.34%, 33.33%, 33.33% -> 33.3%, 33.3%, 33.3%) can leave the
    displayed total at 99.9% or 100.1% due to independent rounding
    error, which looks sloppy in the UI. This guarantees a clean, exact
    100.0% total every time, without changing the underlying model
    output or which class is predicted.
    """
    scaled = probs * 1000  # work in units of 0.1% for 1-decimal precision
    floored = np.floor(scaled).astype(int)
    remainder = 1000 - floored.sum()

    # give the leftover 0.1%-units to whichever classes had the largest
    # fractional part that got cut off by flooring
    fractional_parts = scaled - floored
    order = np.argsort(-fractional_parts)  # largest fractional part first
    for i in range(remainder):
        floored[order[i]] += 1

    return floored / 10.0  # back to percentage with 1 decimal precision


def predict_sentiment(text: str, model, tokenizer, encoder):
    """
    Returns a dict with:
      - prediction: 'positive' | 'negative' | 'neutral'
      - confidence: float 0-100
      - masked_text: text after NER masking (person names replaced)
      - cleaned_text: text after preprocessing (what the model actually saw)
      - probabilities: DataFrame of per-class probabilities
      - is_oov: True if none of the cleaned words were seen during training
    """
    max_len = get_max_len()

    # Step 1: mask person names in the ORIGINAL text (before lowercasing,
    # since capitalization is the main signal the NER model relies on).
    masked_text = mask_person_entities(text)

    # Step 2: normal sentiment cleaning, same as before.
    cleaned_text = clean_text(masked_text)

    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    oov_id = tokenizer.word_index.get(tokenizer.oov_token) if tokenizer.oov_token else None
    real_tokens = [t for t in seq[0] if t != 0 and t != oov_id]

    if not real_tokens:
        probabilities = pd.DataFrame({
            "Sentiment": encoder.classes_,
            "Probability": [100.0 if c == "neutral" else 0.0 for c in encoder.classes_],
        })
        return {
            "prediction": "neutral",
            "confidence": 0.0,
            "masked_text": masked_text,
            "cleaned_text": cleaned_text,
            "probabilities": probabilities,
            "is_oov": True,
        }

    probs = model.predict(padded, verbose=0)[0]
    pred_index = int(np.argmax(probs))
    prediction = encoder.inverse_transform([pred_index])[0]

    clean_probs_pct = _clean_percentages(probs)
    confidence = float(clean_probs_pct[pred_index])

    probabilities = pd.DataFrame({
        "Sentiment": encoder.classes_,
        "Probability": clean_probs_pct,
    })

    return {
        "prediction": prediction,
        "confidence": confidence,
        "masked_text": masked_text,
        "cleaned_text": cleaned_text,
        "probabilities": probabilities,
        "is_oov": False,
    }


def _split_into_chunks(text: str, max_words: int = 30):
    """
    Splits text into sentence-sized chunks close to what the sentiment
    model was actually trained on (median 3 words, and NOT ONE training
    example over 35 words -- see dataset/reddit_sentiment_dataset_v9.csv).
    Feeding a whole multi-paragraph Reddit post through as one long
    sequence is wildly out-of-distribution for this model.
    """
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    chunks = []
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        words = sent.split()
        if len(words) <= max_words:
            chunks.append(sent)
        else:
            # a "sentence" with no punctuation to split on -- hard-split by word count
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i + max_words]))
    return chunks or [text.strip()]


# ---------------------------------------------------------------------
# Real-world complaint/praise phrases that the training set is very
# unlikely to have seen (it's 32,359 short synthetic/templated snippets,
# median 3 words -- see dataset/reddit_sentiment_dataset_v9.csv). Words
# like "stalling", "malfunction", "recurring problem" are probably not
# even in the model's ~10k-word vocabulary, so it has no real signal to
# work with for genuine long-form complaints/reviews. This lexicon is a
# transparent, hand-curated safety net layered on TOP of the model's own
# prediction -- it does not override the model, it adds weighted "votes"
# alongside it, so a post still needs either the model OR clear real-world
# language (ideally both) to land on a class.
# ---------------------------------------------------------------------
NEGATIVE_SIGNAL_PHRASES = {
    # -- product/consumer complaints (original set) --
    "broke down", "broken", "stopped working", "shut off", "shut down",
    "shutting down", "stalled", "stalling", "malfunction", "malfunctioning",
    "defective", "faulty", "not normal", "warning light", "engine light",
    "error message", "crashed", "died again", "failed", "failure",
    "disappointed", "disappointing", "regret", "worst", "terrible",
    "horrible", "unacceptable", "complaint", "refund", "won't start",
    "wont start", "doesn't work", "does not work", "waste of money",
    "scary experience", "recurring problem", "poor quality",
    "bad experience", "never again", "avoid this", "rip off", "ripoff",
    "scam", "cheated", "false promise", "loss of power", "random stalling",
    # -- news/legal/controversy language (added: the product-complaint list
    #    above rarely fires on news articles, courts, protests, disputes,
    #    scandals, etc., which is a very common category of long-form text
    #    fed in through Live News Search) --
    "legal action", "consumer court", "court order", "court directs",
    "paper leak", "exam leak", "cancelled amid", "allegations of",
    "alleged damage", "warns of", "issued a warning", "controversy",
    "backlash", "protest", "protests", "hunger strike", "indefinite fast",
    "chaos", "scrutiny", "setback", "criticism", "criticised", "dispute",
    "lawsuit", "fake", "irregularities", "malpractice", "scandal",
    "compensate", "damaged car", "damaged vehicle", "engine damage",
    "reduce mileage", "reduce efficiency",
}

POSITIVE_SIGNAL_PHRASES = {
    # -- product/consumer praise (original set) --
    "highly recommend", "highly recommended", "love it", "loving it",
    "amazing product", "excellent service", "best purchase", "very happy",
    "extremely happy", "impressed", "works great", "worth it",
    "highly satisfied", "great experience", "exceeded expectations",
    "no complaints", "five stars", "5 stars", "outstanding", "fantastic",
    "superb", "smooth experience", "flawless", "couldn't be happier",
    "very pleased",
    # -- news/general good-news language (added, mirrors the negative
    #    additions above so the lexicon stays balanced rather than only
    #    tilting long-form text toward negative) --
    "record high", "record profit", "successful launch", "praised",
    "praise for", "impressive performance", "improved ranks", "excel",
    "excelled", "shine in", "top rank", "reassure", "reassured",
    "assurance", "clarifications assuring", "counter growing backlash",
}

LEXICON_VOTE_WEIGHT = 12  # how much one matched domain phrase counts, in "words," alongside model chunks
VADER_VOTE_WEIGHT = 1.6   # multiplier on VADER's per-chunk opinion, scaled BY THAT CHUNK'S OWN WORD-COUNT
                          # WEIGHT (see predict_long_text) -- not a flat constant. A flat constant let the
                          # model's own (word-count-scaled) vote drown out VADER on any chunk longer than
                          # ~15 words, which is most real sentences. Scaling VADER by the same per-chunk
                          # weight puts it on equal (and, via the >1 multiplier, slightly favoured) footing
                          # with the model for every chunk length -- important because the model was trained
                          # almost entirely on 1-2 sentence snippets and is a known weaker signal on long,
                          # out-of-distribution text like news articles (its own held-out precision on the
                          # "positive" class is ~78%, i.e. it mislabels real negative/neutral text as
                          # positive noticeably often -- see models/precomputed_metrics_v2.json).


def _vader_vote(chunk_text: str):
    """
    Scores a chunk with VADER (a general-purpose lexicon + rule-based
    sentiment tool built for informal/social-media text -- 7,500+ words,
    handles negation, intensifiers, slang, punctuation/caps emphasis).

    Unlike NEGATIVE_SIGNAL_PHRASES/POSITIVE_SIGNAL_PHRASES above (a small
    hand-picked list that only helps posts using those exact phrases),
    VADER is a broad, pre-built lexicon -- this is what makes the fix
    generalize to other posts/comments/topics, not just this one example.

    Returns (label, strength 0-1). label is "positive", "negative", or
    "neutral" -- VADER's near-zero-compound case is its most confident
    signal that a chunk is genuinely neutral, so it now counts as a real
    vote for the neutral class too (previously it returned (None, 0.0) and
    was thrown away, which meant neutral had NO source of extra evidence
    anywhere in the ensemble -- only positive/negative got lexicon+VADER
    reinforcement -- so a genuinely neutral chunk almost never won against
    the model's own noisy per-chunk guess).
    """
    compound = _vader.polarity_scores(chunk_text)["compound"]
    if compound >= 0.05:
        return "positive", compound
    if compound <= -0.05:
        return "negative", -compound
    # Near-zero compound: strength scales with how close to dead-zero it
    # is (1.0 exactly at compound=0, tapering to 0 at the +/-0.05 edges).
    return "neutral", 1.0 - (abs(compound) / 0.05)


def predict_long_text(text: str, model, tokenizer, encoder, max_chunk_words: int = 30):
    """
    Sentiment for long-form text (full Reddit posts/comments spanning
    several sentences), by scoring each sentence-sized chunk with
    predict_sentiment() individually and combining the results --
    instead of feeding the entire post through as one giant sequence.

    Three things make the "one giant sequence" approach unreliable for
    long, real-world posts specifically:
      1. The model's training data is almost entirely 1-2 sentence
         snippets (median 3 words, max 35 words ever seen in training).
      2. max_len=40 truncates anything longer anyway, silently dropping
         everything past the first ~40 tokens.
      3. Real complaint/praise vocabulary is likely missing from the
         model's vocabulary entirely, and a bias toward certain repeated
         words (e.g. brand/product names) can show up across MANY
         sentences of the same post, not just one -- so per-chunk
         scoring alone can still be outvoted by that bias.

    This function addresses (1) and (2) by chunking + a length-weighted
    average, and (3) by blending in two extra signals alongside the
    model's own chunk predictions:
      - NEGATIVE_SIGNAL_PHRASES / POSITIVE_SIGNAL_PHRASES: a small,
        hand-curated list for a few very common complaint/praise phrases.
      - VADER (_vader_vote): a broad, general-purpose sentiment lexicon
        that generalizes across topics/wording, not just the specific
        phrases above -- this is what makes the fix work for OTHER
        long posts and comments too, not only the one it was tested on.

    None of these override the model -- they add weighted votes to the
    same aggregation, so the final verdict reflects all three sources
    together.

    Returns the same shape as predict_sentiment(), plus:
      - chunk_results: list of per-chunk (text, prediction, confidence),
        useful for showing "why" a long post got its overall verdict.
    """
    chunks = _split_into_chunks(text, max_chunk_words)

    chunk_results = []
    for chunk in chunks:
        result = predict_sentiment(chunk, model, tokenizer, encoder)
        if result["is_oov"]:
            continue  # no recognized vocabulary in this chunk at all -- skip it
        chunk_results.append((chunk, result))

    if not chunk_results:
        # Nothing usable in any chunk -- fall back to the plain whole-text
        # prediction so callers still get a result rather than nothing.
        fallback = predict_sentiment(text, model, tokenizer, encoder)
        fallback["chunk_results"] = []
        return fallback

    # Weight each chunk's contribution by its word count, so a throwaway
    # short chunk doesn't outweigh a substantial one. Lexicon phrase
    # matches and VADER's opinion add extra weighted votes on top of this.
    labels = list(encoder.classes_)
    weighted_probs = {label: 0.0 for label in labels}
    total_weight = 0.0

    for chunk, result in chunk_results:
        weight = max(len(chunk.split()), 1)
        total_weight += weight
        for _, row in result["probabilities"].iterrows():
            weighted_probs[row["Sentiment"]] += row["Probability"] * weight

        chunk_lower = chunk.lower()
        neg_hits = sum(1 for phrase in NEGATIVE_SIGNAL_PHRASES if phrase in chunk_lower)
        pos_hits = sum(1 for phrase in POSITIVE_SIGNAL_PHRASES if phrase in chunk_lower)
        if neg_hits and "negative" in weighted_probs:
            bonus = neg_hits * LEXICON_VOTE_WEIGHT
            weighted_probs["negative"] += 100.0 * bonus
            total_weight += bonus
        if pos_hits and "positive" in weighted_probs:
            bonus = pos_hits * LEXICON_VOTE_WEIGHT
            weighted_probs["positive"] += 100.0 * bonus
            total_weight += bonus

        vader_label, vader_strength = _vader_vote(chunk)
        if vader_label and vader_label in weighted_probs and vader_strength > 0:
            # Scaled by `weight` (this chunk's own word count) so VADER's say is
            # proportional to the model's say on the SAME chunk, not a flat
            # constant that only mattered for short chunks. See VADER_VOTE_WEIGHT
            # comment above for why.
            bonus = weight * vader_strength * VADER_VOTE_WEIGHT
            weighted_probs[vader_label] += 100.0 * bonus
            total_weight += bonus

    for label in labels:
        weighted_probs[label] /= total_weight

    probabilities = pd.DataFrame({
        "Sentiment": labels,
        "Probability": [weighted_probs[l] for l in labels],
    })

    pred_label = max(weighted_probs, key=weighted_probs.get)
    confidence = weighted_probs[pred_label]

    return {
        "prediction": pred_label,
        "confidence": confidence,
        "masked_text": " | ".join(r["masked_text"] for _, r in chunk_results),
        "cleaned_text": " | ".join(r["cleaned_text"] for _, r in chunk_results),
        "probabilities": probabilities,
        "is_oov": False,
        "chunk_results": [
            {"chunk": c, "prediction": r["prediction"], "confidence": round(r["confidence"], 1)}
            for c, r in chunk_results
        ],
    }


def get_top_contributing_words(cleaned_text: str, tokenizer, top_n: int = 8):
    """
    With an LSTM there's no single per-word weight like TF-IDF had, since
    the model's decision depends on word ORDER and context, not just
    which words are present. As an honest, useful proxy, this scores
    each known word by how rare it was in the training vocabulary
    (its rank in tokenizer.word_index, scaled to 0-100) -- rarer, more
    specific words tend to carry more sentiment signal than very common
    ones, which is the same intuition TF-IDF was built on.
    """
    if not cleaned_text.strip():
        return []

    words = cleaned_text.split()
    vocab_size = len(tokenizer.word_index) or 1
    known = [(w, tokenizer.word_index[w]) for w in words if w in tokenizer.word_index]
    if not known:
        return []

    scored = [(w, round(rank / vocab_size * 100, 1)) for w, rank in known]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

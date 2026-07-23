"""
utils/predictor.py  (v3 -- with NER masking)
-----------------------------------------------
Wraps the full pipeline: NER masking -> text cleaning -> tokenizing ->
padding -> LSTM sentiment prediction.

New in this version: before any lowercasing/cleaning happens, the raw
text is passed through the trained NER model to detect and mask person
names (e.g. "Happy Singh is sad" -> "person is sad"), so the sentiment
model doesn't confuse a name like "Happy" with the emotion word "happy".

============================================================================
HEAVILY COMMENTED VERSION -- every block below has a plain-language note
explaining WHAT the code does and WHY, added for easier reading. No logic
was changed from the original file -- only comments were added/expanded.
============================================================================
"""

import re                      # regex -- used later to split long text into sentences
import nltk                    # NLTK -- gives us VADER (a ready-made sentiment lexicon)
import numpy as np             # numpy -- array math (probabilities, rounding, sorting)
import pandas as pd            # pandas -- used to build the "probabilities" table returned to callers
from nltk.sentiment import SentimentIntensityAnalyzer          # VADER's actual analyzer class
from tensorflow.keras.preprocessing.sequence import pad_sequences  # pads/truncates token sequences to a fixed length

# Our own project helpers (from other files in utils/):
from utils.preprocessing import clean_text     # lowercases + strips punctuation/stopwords etc.
from utils.loader import get_max_len           # returns the fixed sequence length the LSTM was trained on
from utils.ner import mask_person_entities     # replaces detected person names with the word "person"


def _ensure_vader_resource() -> None:
    """Downloads VADER's lexicon data only if not already present (same pattern as
    ensure_nltk_resources() in preprocessing.py for stopwords)."""
    try:
        # nltk.data.find() looks on disk for the VADER lexicon file.
        # If it's already downloaded, this line succeeds and we do nothing further.
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        # If the file isn't found, nltk.data.find() raises LookupError --
        # in that case, download it now (one-time cost, first run only).
        nltk.download("vader_lexicon")


# ---- Module-load-time setup (runs ONCE, the moment this file is imported) ----
_ensure_vader_resource()                 # make sure the VADER lexicon exists on disk before we try to use it
_vader = SentimentIntensityAnalyzer()    # build one VADER object and reuse it everywhere (avoids re-creating it on every prediction call)


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
    # Step 1: work in units of "0.1%" instead of raw fractions (0-1 range),
    # so that after we floor to whole numbers, each unit represents 0.1%.
    # Example: probs = [0.334, 0.333, 0.333] -> scaled = [334.0, 333.0, 333.0]
    scaled = probs * 1000  # work in units of 0.1% for 1-decimal precision

    # Step 2: floor (round DOWN) each value to the nearest whole number.
    # This is where "leftover" fractional bits get chopped off and lost.
    floored = np.floor(scaled).astype(int)

    # Step 3: figure out how many 0.1%-units were lost in total by flooring.
    # Since scaled always sums to exactly 1000, comparing floored.sum() to
    # 1000 tells us exactly how many whole units still need to be given back.
    remainder = 1000 - floored.sum()

    # give the leftover 0.1%-units to whichever classes had the largest
    # fractional part that got cut off by flooring
    # Step 4: work out how big a "cut" each class took when we floored it.
    fractional_parts = scaled - floored

    # Step 5: sort classes by fractional part, LARGEST first.
    # np.argsort sorts ascending by default, so we negate the values to flip
    # it into descending order -- the class that lost the most decimal value
    # (i.e. was closest to rounding UP) ends up first in this list.
    order = np.argsort(-fractional_parts)  # largest fractional part first

    # Step 6: hand back exactly `remainder` units, one each, to the classes
    # that "deserve" it most (the ones with the biggest fractional parts),
    # in priority order. This guarantees the total adds back up to 1000.
    for i in range(remainder):
        floored[order[i]] += 1

    # Step 7: convert back from "0.1%-units" to normal percentage (1 decimal place).
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
    # The fixed sequence length the LSTM expects as input (set during training).
    max_len = get_max_len()

    # Step 1: mask person names in the ORIGINAL text (before lowercasing,
    # since capitalization is the main signal the NER model relies on).
    # Example: "Happy Singh is sad" -> "person is sad"
    masked_text = mask_person_entities(text)

    # Step 2: normal sentiment cleaning, same as before.
    # (lowercasing, punctuation/stopword removal, etc. -- see preprocessing.py)
    cleaned_text = clean_text(masked_text)

    # Step 3: convert the cleaned text into a sequence of integer token IDs
    # using the vocabulary the tokenizer learned during training.
    seq = tokenizer.texts_to_sequences([cleaned_text])

    # Step 4: pad (with zeros) or truncate the sequence so it's exactly
    # `max_len` tokens long -- the LSTM requires a fixed-size input.
    # padding="post" / truncating="post" means zeros/cuts happen at the END.
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    # Step 5: figure out which token ID represents "unknown word" (OOV),
    # so we can tell apart real recognized words from padding (0) and OOV.
    oov_id = tokenizer.word_index.get(tokenizer.oov_token) if tokenizer.oov_token else None

    # real_tokens = only the tokens that are neither padding (0) nor OOV --
    # i.e. words the model actually has real training signal for.
    real_tokens = [t for t in seq[0] if t != 0 and t != oov_id]

    if not real_tokens:
        # Nothing recognizable in this text at all (every word was unknown
        # or the text was empty). Rather than feeding pure noise into the
        # model, short-circuit and call it "neutral" with 0% confidence.
        probabilities = pd.DataFrame({
            "Sentiment": encoder.classes_,
            # Put 100% on "neutral" and 0% on everything else, just so the
            # returned table has a valid, sane shape for the caller/UI.
            "Probability": [100.0 if c == "neutral" else 0.0 for c in encoder.classes_],
        })
        return {
            "prediction": "neutral",
            "confidence": 0.0,
            "masked_text": masked_text,
            "cleaned_text": cleaned_text,
            "probabilities": probabilities,
            "is_oov": True,   # flag so callers know this was a "no real signal" case
        }

    # Step 6: run the actual LSTM model on the padded sequence.
    # model.predict returns probabilities for each class; [0] because we
    # only passed one example, so we grab the first (only) row of results.
    probs = model.predict(padded, verbose=0)[0]

    # Step 7: find which class index has the highest probability.
    pred_index = int(np.argmax(probs))

    # Step 8: convert that numeric class index back into a human label
    # ("positive"/"negative"/"neutral") using the same encoder used in training.
    prediction = encoder.inverse_transform([pred_index])[0]

    # Step 9: clean up the raw probabilities into nice percentages that sum to 100.0
    clean_probs_pct = _clean_percentages(probs)
    confidence = float(clean_probs_pct[pred_index])   # confidence = % for the predicted class only

    # Step 10: build a small table (Sentiment | Probability) for display purposes.
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
    # Split the text into "sentences" using a regex:
    #   (?<=[.!?])\s+   -> a space (or spaces) that comes right AFTER . ! or ?
    #   |\n+            -> OR one-or-more newlines
    # This is a simple sentence-boundary splitter (not perfect grammar-aware,
    # but good enough for chunking purposes).
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())

    chunks = []
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue   # skip empty strings (can happen from splitting)

        words = sent.split()

        if len(words) <= max_words:
            # Sentence is already short enough -- keep it as one chunk.
            chunks.append(sent)
        else:
            # a "sentence" with no punctuation to split on -- hard-split by word count
            # (e.g. a huge run-on sentence, or a whole paragraph with no periods).
            # We slice it into fixed-size word groups of `max_words` each.
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i + max_words]))

    # Safety fallback: if for some reason nothing got added to `chunks`
    # (e.g. text was just whitespace/punctuation), return the whole
    # original text as a single chunk instead of an empty list.
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

# NOTE: These are plain Python `set`s of phrases. A `set` is used (instead
# of a list) purely for fast membership checks (`phrase in some_set`).
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

# How much a single matched lexicon phrase "counts for", expressed in the
# same units as a chunk's word-count weight. E.g. LEXICON_VOTE_WEIGHT = 12
# means: one matched phrase in a chunk pulls as much weight toward its
# class as a 12-word chunk of plain model output would.
LEXICON_VOTE_WEIGHT = 12  # how much one matched domain phrase counts, in "words," alongside model chunks

# Multiplier applied to VADER's per-chunk vote. NOT a flat constant --
# it's applied on top of the chunk's own word-count weight (see
# predict_long_text below), so VADER's influence scales up/down together
# with the model's influence for that same chunk, instead of staying fixed
# while the model's vote grows with chunk length.
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
    # VADER's "compound" score is a single number from -1 (most negative)
    # to +1 (most positive), summarizing the whole chunk's sentiment.
    compound = _vader.polarity_scores(chunk_text)["compound"]

    if compound >= 0.05:
        # Clearly positive by VADER's own convention (>= 0.05 threshold).
        # Strength = the compound score itself (bigger = more confident).
        return "positive", compound

    if compound <= -0.05:
        # Clearly negative. Strength = absolute value (so it's always positive/0-1-ish).
        return "negative", -compound

    # Near-zero compound: strength scales with how close to dead-zero it
    # is (1.0 exactly at compound=0, tapering to 0 at the +/-0.05 edges).
    # i.e. compound == 0   -> strength = 1.0 (VADER is very sure it's neutral)
    #      compound == 0.05 or -0.05 -> strength = 0.0 (right at the edge, no real signal)
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
    # Step 1: break the long text into small, model-friendly chunks.
    chunks = _split_into_chunks(text, max_chunk_words)

    chunk_results = []
    for chunk in chunks:
        # Step 2: run the normal single-text pipeline on EACH chunk individually.
        result = predict_sentiment(chunk, model, tokenizer, encoder)

        if result["is_oov"]:
            # This chunk had literally no recognizable vocabulary at all --
            # skip it entirely rather than let a meaningless "neutral 0%"
            # guess dilute the overall vote.
            continue

        chunk_results.append((chunk, result))

    if not chunk_results:
        # Every single chunk was OOV/unusable -- there's nothing to vote
        # with. Fall back to running the plain whole-text prediction so the
        # caller still gets SOME result instead of an empty/broken response.
        fallback = predict_sentiment(text, model, tokenizer, encoder)
        fallback["chunk_results"] = []
        return fallback

    # Weight each chunk's contribution by its word count, so a throwaway
    # short chunk doesn't outweigh a substantial one. Lexicon phrase
    # matches and VADER's opinion add extra weighted votes on top of this.
    labels = list(encoder.classes_)                       # e.g. ["negative", "neutral", "positive"]
    weighted_probs = {label: 0.0 for label in labels}      # running weighted total per class, starts at 0
    total_weight = 0.0                                     # running total of ALL weight added (for normalizing later)

    for chunk, result in chunk_results:
        # --- (A) Base vote: the model's own per-chunk prediction ---
        # Longer chunks get proportionally more "say" in the final vote
        # (a 20-word chunk matters more than a 3-word chunk).
        weight = max(len(chunk.split()), 1)   # word count of this chunk (at least 1, to avoid zero-weight)
        total_weight += weight

        # Add this chunk's own probability table into the running totals,
        # scaled by its weight.
        for _, row in result["probabilities"].iterrows():
            weighted_probs[row["Sentiment"]] += row["Probability"] * weight

        # --- (B) Extra vote: hand-curated phrase lists ---
        chunk_lower = chunk.lower()
        # Count how many negative/positive phrases from our lists appear in this chunk.
        neg_hits = sum(1 for phrase in NEGATIVE_SIGNAL_PHRASES if phrase in chunk_lower)
        pos_hits = sum(1 for phrase in POSITIVE_SIGNAL_PHRASES if phrase in chunk_lower)

        if neg_hits and "negative" in weighted_probs:
            # Each matched phrase acts like a mini "100%-confident vote"
            # worth LEXICON_VOTE_WEIGHT words, added directly to "negative".
            bonus = neg_hits * LEXICON_VOTE_WEIGHT
            weighted_probs["negative"] += 100.0 * bonus
            total_weight += bonus   # must also add to total_weight so the final normalization stays correct

        if pos_hits and "positive" in weighted_probs:
            bonus = pos_hits * LEXICON_VOTE_WEIGHT
            weighted_probs["positive"] += 100.0 * bonus
            total_weight += bonus

        # --- (C) Extra vote: VADER's opinion on this same chunk ---
        vader_label, vader_strength = _vader_vote(chunk)
        if vader_label and vader_label in weighted_probs and vader_strength > 0:
            # Scaled by `weight` (this chunk's own word count) so VADER's say is
            # proportional to the model's say on the SAME chunk, not a flat
            # constant that only mattered for short chunks. See VADER_VOTE_WEIGHT
            # comment above for why.
            bonus = weight * vader_strength * VADER_VOTE_WEIGHT
            weighted_probs[vader_label] += 100.0 * bonus
            total_weight += bonus

    # Step 3: normalize -- divide every class's accumulated weighted total
    # by the grand total weight, turning raw sums back into 0-100 percentages.
    for label in labels:
        weighted_probs[label] /= total_weight

    # Step 4: build the final probabilities table for display.
    probabilities = pd.DataFrame({
        "Sentiment": labels,
        "Probability": [weighted_probs[l] for l in labels],
    })

    # Step 5: the winning class is whichever has the highest weighted percentage.
    pred_label = max(weighted_probs, key=weighted_probs.get)
    confidence = weighted_probs[pred_label]

    return {
        "prediction": pred_label,
        "confidence": confidence,
        # Combine every chunk's own masked/cleaned text into one string,
        # separated by " | ", so the caller can see what was actually fed
        # into the model across all chunks.
        "masked_text": " | ".join(r["masked_text"] for _, r in chunk_results),
        "cleaned_text": " | ".join(r["cleaned_text"] for _, r in chunk_results),
        "probabilities": probabilities,
        "is_oov": False,
        # Per-chunk breakdown -- lets the UI show WHY the overall post got
        # its final verdict (e.g. "3 of 5 chunks were negative").
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
        # Nothing to score if the text is empty/whitespace.
        return []

    words = cleaned_text.split()
    # tokenizer.word_index maps word -> rank (1 = most frequent word in
    # training data, higher numbers = rarer words). Guard against a
    # tokenizer with an empty vocabulary (avoid divide-by-zero later).
    vocab_size = len(tokenizer.word_index) or 1

    # Keep only words the tokenizer actually recognizes, paired with their rank.
    known = [(w, tokenizer.word_index[w]) for w in words if w in tokenizer.word_index]
    if not known:
        return []

    # Score each word: higher rank number (rarer word) -> higher score (0-100).
    # This is why it's called a "proxy" for importance -- rarity is used as
    # a stand-in for how much sentiment signal a word likely carries.
    scored = [(w, round(rank / vocab_size * 100, 1)) for w, rank in known]

    # Sort so the rarest/most "important" words come first.
    scored.sort(key=lambda x: x[1], reverse=True)

    # Return only the top N words.
    return scored[:top_n]

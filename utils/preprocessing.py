"""
utils/preprocessing.py  (v2)
------------------------------
Text cleaning logic. This version PRESERVES negation and contrast words
("not", "no", "never", "but", "however"...) instead of stripping them as
generic stopwords, because the v2 model is a sequence model (LSTM) that
relies on word order and these specific words to understand context
(e.g. "good day BUT tired", "not good").

IMPORTANT: this must stay identical to the cleaning function used in the
v2 training notebook (Reddit_Sentiment_LSTM_Training_v2.ipynb), otherwise
predictions at inference time won't match what the model was trained on.
"""

import re
import nltk
from nltk.corpus import stopwords


def ensure_nltk_resources() -> None:
    """Downloads required NLTK data only if not already present."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


ensure_nltk_resources()

# Words that carry negation or contrast meaning -- deliberately KEPT,
# even though NLTK's default stopword list would normally remove them.
NEGATION_AND_CONTRAST_WORDS = {
    "not", "no", "nor", "never",
    "but", "however", "although", "though", "yet", "still",
    "don't", "doesn't", "didn't", "isn't", "wasn't", "aren't", "weren't",
    "won't", "wouldn't", "can't", "cannot", "couldn't", "shouldn't",
    "hasn't", "haven't", "hadn't",
    "dont", "doesnt", "didnt", "isnt", "wasnt", "arent", "werent",
    "wont", "wouldnt", "cant", "couldnt", "shouldnt", "hasnt", "havent", "hadnt",
}

_ALL_STOPWORDS = set(stopwords.words("english"))
_STOP_WORDS = _ALL_STOPWORDS - NEGATION_AND_CONTRAST_WORDS


def clean_text(text: str) -> str:
    """
    Cleans raw text the same way the v2 training notebook did:
    lowercase -> strip URLs -> strip non-letters (keep apostrophes) ->
    tokenize -> remove stopwords EXCEPT negation/contrast words.

    Word order is preserved (never sorted or bagged) because this feeds
    a sequence model (LSTM) that reads words in order.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z'\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in _STOP_WORDS]
    return " ".join(words)

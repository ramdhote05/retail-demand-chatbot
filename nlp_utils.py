"""
nlp_utils.py
------------
Text preprocessing utilities for the chatbot: lowercasing, punctuation
stripping, tokenizing, stopword removal, and WordNet lemmatization.

This is the step that turns messy user input like:
    "Hellooo!! Where IS my package??"
into a normalized form like:
    "hello package"
so it can be compared fairly against the chatbot's known patterns.
"""

import re
import nltk

# ---------------------------------------------------------------------------
# Make sure the NLTK data we need is available. This only actually downloads
# anything the first time the script runs on a machine (requires internet
# access once); after that it's cached locally and these calls are no-ops.
# ---------------------------------------------------------------------------
_NLTK_PACKAGES = ["punkt", "punkt_tab", "wordnet", "omw-1.4", "stopwords"]

for _package in _NLTK_PACKAGES:
    try:
        nltk.download(_package, quiet=True)
    except Exception:
        # Some package names (e.g. punkt_tab) don't exist on older NLTK
        # versions — that's fine, the older "punkt" package covers it.
        pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_lemmatizer = WordNetLemmatizer()

# Keep negation words — "not", "can't", etc. — because they flip meaning
# and we don't want them stripped out as generic stopwords.
_NEGATIONS = {"no", "not", "nor", "cannot", "cant", "dont", "wont", "isnt", "arent"}
_STOPWORDS = set(stopwords.words("english")) - _NEGATIONS


def clean_text(text: str) -> str:
    """Lowercase and strip everything except letters, numbers, and apostrophes."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline: clean -> tokenize -> remove stopwords
    -> lemmatize. Returns a single space-joined string ready to be fed
    into the TF-IDF vectorizer.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    tokens = word_tokenize(cleaned)
    lemmas = [
        _lemmatizer.lemmatize(token)
        for token in tokens
        if token not in _STOPWORDS and token.strip()
    ]

    # If everything got filtered out as stopwords (e.g. input was just
    # "is it"), fall back to the cleaned-but-unfiltered text so we still
    # have something to compare.
    return " ".join(lemmas) if lemmas else cleaned


if __name__ == "__main__":
    samples = [
        "Hellooo!! Where IS my package??",
        "What's your return policy?",
        "I cannot log into my account",
    ]
    for s in samples:
        print(f"{s!r:45} -> {preprocess(s)!r}")

# Simple AI Chatbot

A rule-based chatbot built with NLTK and scikit-learn, with a Streamlit chat UI. It normalizes user input (tokenizing + WordNet lemmatization), then matches it against predefined intents using TF-IDF vectorization and cosine similarity.

## Files

- `intents.json` — the bot's knowledge base: each intent has example `patterns` (things a user might say) and possible `responses`.
- `nlp_utils.py` — text preprocessing: cleaning, tokenizing, stopword removal, lemmatization.
- `chatbot_engine.py` — the matching logic (TF-IDF + cosine similarity). No UI code at all, so it's easy to test or reuse.
- `app.py` — the Streamlit front-end.
- `requirements.txt` — Python dependencies.

## Setup

It's a good idea to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first time you run the bot, NLTK will download a few small data packages (tokenizer models, WordNet, stopwords) automatically — this needs internet access once, then it's cached locally.

## Running it

Quickest way to test the logic in your terminal, no Streamlit needed:

```bash
python3 chatbot_engine.py
```

To launch the actual chat UI:

```bash
streamlit run app.py
```

This opens a browser tab with a chat window. There's a sidebar with example topics, a "Clear conversation" button, and a debug toggle that shows which intent your last message matched and the confidence score — handy while you're tuning things.

## How the matching works

Every example pattern from every intent gets preprocessed and combined into one corpus, and a single `TfidfVectorizer` is fit on it. When you type a message, it's preprocessed the same way, vectorized, and compared against every pattern with cosine similarity. The intent attached to the closest pattern wins, as long as the similarity score clears `CONFIDENCE_THRESHOLD` (set to 0.4 in `chatbot_engine.py`) — otherwise the bot falls back to its `fallback` responses rather than guessing confidently at something irrelevant.

## Extending it

To teach the bot something new, just add an entry to `intents.json`:

```json
{
  "tag": "store_locations",
  "patterns": ["where is your store", "do you have a physical location", "store address"],
  "responses": ["We're online-only right now, but check back for store openings!"]
}
```

More example patterns per intent generally means better matching — five or six varied phrasings per intent is a reasonable target.

## A known limitation worth knowing about

TF-IDF + cosine similarity on a small set of short patterns can occasionally misfire on out-of-scope questions, matching them to an unrelated intent just because they share a common word (e.g. "what" or "tell me"). This is a real characteristic of the technique, not a bug, and it's a good thing to notice while experimenting: with only a handful of patterns per intent, rare shared words can dominate the similarity score. Adding more patterns per intent and raising `CONFIDENCE_THRESHOLD` both help, but a small rule-based bot like this will never be bulletproof against questions far outside its intents. That gap is exactly the kind of thing more advanced approaches (real intent classifiers, embeddings, or LLM-based bots) are built to close.

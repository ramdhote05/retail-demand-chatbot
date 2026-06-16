"""
chatbot_engine.py
------------------
Core chatbot logic. This module has no UI dependencies at all (no
Streamlit, no Tkinter) so it can be tested directly from the command
line — see the __main__ block at the bottom — or imported by any
front-end you like.

How matching works:
1. Every example "pattern" from every intent in intents.json gets
   preprocessed (nlp_utils.preprocess) and combined into one corpus.
2. A single TfidfVectorizer is fit on that whole corpus.
3. When the user types something, we preprocess it the same way,
   vectorize it, and compute cosine similarity against every pattern.
4. The intent of the closest-matching pattern wins — as long as the
   similarity score clears CONFIDENCE_THRESHOLD. Otherwise we fall
   back to the "fallback" intent so the bot doesn't confidently say
   something irrelevant.
"""

import json
import random
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp_utils import preprocess

# Minimum cosine similarity required to accept a match. Lower this to
# make the bot more willing to guess; raise it to make it stricter.
# 0.0 = always matches something, 1.0 = requires an exact match.
CONFIDENCE_THRESHOLD = 0.4


class ChatbotEngine:
    def __init__(self, intents_path: str = "intents.json"):
        self.intents_path = Path(intents_path)
        self.intents = self._load_intents()
        self._build_vectorizer()
        self.history = []  # (user_text, bot_response, matched_tag, score)

    # -- setup -------------------------------------------------------

    def _load_intents(self):
        with open(self.intents_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["intents"]

    def _build_vectorizer(self):
        self.corpus = []       # preprocessed pattern text, one entry per pattern
        self.corpus_tags = []  # the intent tag each corpus entry belongs to

        for intent in self.intents:
            for pattern in intent.get("patterns", []):
                processed = preprocess(pattern)
                if processed:
                    self.corpus.append(processed)
                    self.corpus_tags.append(intent["tag"])

        if not self.corpus:
            raise ValueError("No patterns found in intents.json — nothing to match against.")

        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    # -- matching ------------------------------------------------------

    def _match_intent(self, user_text: str):
        cleaned = preprocess(user_text)
        if not cleaned:
            return "fallback", 0.0

        user_vector = self.vectorizer.transform([cleaned])
        similarities = cosine_similarity(user_vector, self.tfidf_matrix)[0]

        best_index = similarities.argmax()
        best_score = float(similarities[best_index])

        if best_score < CONFIDENCE_THRESHOLD:
            return "fallback", best_score

        return self.corpus_tags[best_index], best_score

    def _responses_for_tag(self, tag: str):
        for intent in self.intents:
            if intent["tag"] == tag:
                return intent["responses"]
        # Tag not found at all (shouldn't normally happen) — use fallback.
        return self._responses_for_tag("fallback")

    # -- public API ----------------------------------------------------

    def get_response(self, user_text: str) -> str:
        """Takes raw user text, returns the bot's reply as a string."""
        tag, score = self._match_intent(user_text)
        response = random.choice(self._responses_for_tag(tag))
        self.history.append((user_text, response, tag, round(score, 3)))
        return response


if __name__ == "__main__":
    # Lets you test the chatbot logic in a plain terminal, with no
    # Streamlit or Tkinter needed — useful while you're still tweaking
    # intents.json.
    intents_file = Path(__file__).parent / "intents.json"
    bot = ChatbotEngine(intents_path=str(intents_file))

    print("Simple AI Chatbot (CLI test mode). Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bot: Goodbye!")
            break
        if not user_input:
            continue
        reply = bot.get_response(user_input)
        tag, score = bot.history[-1][2], bot.history[-1][3]
        print(f"Bot: {reply}   [matched: {tag}, score: {score}]")

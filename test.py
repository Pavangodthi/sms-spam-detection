"""
test.py
========

This script demonstrates how to perform inference using the trained
logistic‑regression model and vectoriser saved by ``train.py``.  It is a
replacement for the original Keras‑based test notebook, allowing users to
classify new SMS messages without requiring TensorFlow/Keras.  The script
loads ``logistic_model.pkl`` and ``cv.pkl`` from the current working
directory, transforms user‑provided text messages into the vector space and
outputs whether each message is predicted to be ``spam`` or ``ham``.

Usage:

.. code:: bash

    python test.py "Your message here" ["Another message" ...]

If no messages are supplied on the command line, the script falls back to
classifying two example messages:

* A promotional message, expected to be identified as spam.
* A simple greeting, expected to be identified as ham.
"""

import pickle
import re
import sys
from typing import List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from nltk.stem.porter import PorterStemmer
except ImportError:
    # Fallback no‑op stemmer if nltk is unavailable
    class PorterStemmer:  # type: ignore
        def stem(self, word: str) -> str:
            return word


def clean_message(msg: str) -> str:
    """Apply the same preprocessing used during training to a single message."""
    stop_words = set(ENGLISH_STOP_WORDS)
    ps = PorterStemmer()
    words = re.sub("[^a-zA-Z]", " ", msg).lower().split()
    tokens = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(tokens)


def classify_messages(messages: List[str]) -> None:
    """Load the model and vectoriser and classify each message in ``messages``.

    Results are printed to stdout.
    """
    # Load vectoriser and logistic model
    try:
        with open("cv.pkl", "rb") as f:
            vectoriser = pickle.load(f)
        with open("logistic_model.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(
            "Error: required model files not found.  Please run 'python train.py' first to generate cv.pkl and logistic_model.pkl."
        )
        sys.exit(1)

    for text in messages:
        cleaned = clean_message(text)
        x_vec = vectoriser.transform([cleaned]).toarray()
        pred = model.predict(x_vec)[0]
        label = "spam" if pred == 1 else "ham"
        print(f"Input: '{text}' → Prediction: {label}")


def main() -> None:
    # Accept messages from command‑line arguments or fall back to examples
    if len(sys.argv) > 1:
        msgs = sys.argv[1:]
    else:
        msgs = [
            "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18",
            "hi how are you",
        ]
    classify_messages(msgs)


if __name__ == "__main__":
    main()
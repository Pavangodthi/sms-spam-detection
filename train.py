"""
train.py
==========

This script provides a simple command‑line interface for training a spam/ham
classifier on the SMS Spam Collection dataset.  It mirrors the original
notebook's preprocessing steps (lower‑casing, removing non‑alphabetic
characters and stop‑words, and stemming) but replaces the Keras neural
network with a logistic‑regression model from scikit‑learn.  This makes the
project runnable in environments where TensorFlow/Keras are not installed.

When run, the script reads ``spam.csv`` from the current working directory,
cleans and vectorises the messages using a ``CountVectorizer`` (limited to
the 100 most frequent tokens), splits the data into a training and test
portion, trains a ``LogisticRegression`` model, reports accuracy on the test
set and writes the fitted vectoriser and classifier to disk.  The fitted
vectoriser is saved as ``cv.pkl`` and the classifier as ``logistic_model.pkl``.

Usage:

.. code:: bash

    python train.py

The script prints the achieved accuracy to standard output and produces the
pickle files in the working directory.  The resulting model can be used
with ``test.py`` to classify new SMS messages without re‑training.
"""

import os
import pickle
import re
from typing import List

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    from nltk.stem.porter import PorterStemmer
except ImportError:
    # NLTK is not strictly necessary for the model to work.  If it's not
    # available, we fall back to a no‑op stemmer.
    class PorterStemmer:  # type: ignore
        def stem(self, word: str) -> str:
            return word


def clean_messages(messages: List[str]) -> List[str]:
    """Return a list of cleaned SMS messages.

    Each message is converted to lowercase, punctuation and numbers are
    removed, stop‑words are filtered out and the remaining tokens are
    stemmed.  A fallback no‑op stemmer is used if NLTK is not installed.

    Parameters
    ----------
    messages: list of str
        Raw SMS messages to be cleaned.

    Returns
    -------
    list of str
        Cleaned and tokenised messages joined back into strings.
    """
    stop_words = set(ENGLISH_STOP_WORDS)
    ps = PorterStemmer()
    cleaned = []
    for msg in messages:
        # remove non letters and lower‑case
        words = re.sub("[^a-zA-Z]", " ", msg).lower().split()
        # filter stopwords and apply stemming
        tokens = [ps.stem(w) for w in words if w not in stop_words]
        cleaned.append(" ".join(tokens))
    return cleaned


def main() -> None:
    # Load the dataset; only the first two columns (label and message) are used
    path = "spam.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset '{path}' not found.  Make sure spam.csv is present in the working directory."
        )
    df = pd.read_csv(path, encoding="latin-1").iloc[:, :2]
    df.columns = ["label", "message"]

    # Preprocess messages and encode labels
    processed = clean_messages(df["message"].tolist())
    y = df["label"].replace({"ham": 0, "spam": 1}).values

    # Vectorise messages
    vectoriser = CountVectorizer(max_features=100)
    X = vectoriser.fit_transform(processed).toarray()

    # Split into train/test and train the logistic regression model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    # Evaluate on held‑out test set
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Logistic Regression accuracy: {accuracy:.4f}")

    # Persist the vectoriser and model for later reuse
    with open("cv.pkl", "wb") as f:
        pickle.dump(vectoriser, f)
    with open("logistic_model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
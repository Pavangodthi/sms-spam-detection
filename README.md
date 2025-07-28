# SMS Spam Detection

This repository demonstrates a simple SMS spam/ham classifier.

## Dataset

The model is trained on the widely used SMS Spam Collection dataset, which contains 5,572 English SMS messages labeled as `ham` (legitimate) or `spam`. A copy of the dataset is included in **`spam.csv`** (columns beyond the first two are removed).

## Training process

Historically, this project used a small Keras neural network. However, to make the project runnable in environments without TensorFlow, the current version uses scikit‑learn instead. The training logic now lives in **`train.py`** and performs the following steps:

1. **Load and clean the data** – Messages are loaded from `spam.csv`, converted to lowercase, non‑alphabetic characters are removed, stop‑words are filtered using scikit‑learn’s built‑in stop‑word list and the remaining words are stemmed (if NLTK is available).

2. **Vectorize** – A `CountVectorizer` converts the cleaned text to a bag‑of‑words representation using the top 100 tokens. The fitted vectorizer is saved to **`cv.pkl`** so it can be reused for inference.

3. **Model architecture** – A `LogisticRegression` classifier from scikit‑learn is used instead of a neural network. This choice avoids the need for TensorFlow/Keras while still providing competitive performance on the SMS spam dataset.

4. **Training** – The classifier is trained on 80 % of the data and evaluated on the remaining 20 %. Accuracy on the hold‑out set is printed when `train.py` is run. After training, the classifier is saved to **`logistic_model.pkl`**.

## Testing / inference

Use the **`test.py`** script to classify new SMS messages. It loads `logistic_model.pkl` and `cv.pkl`, cleans each message in the same way as during training, vectorizes it and outputs whether it is labelled as `spam` or `ham`. If you provide messages as command‑line arguments, they will be classified; otherwise, two default messages are used.

Example:

```bash
python train.py   # trains the model and saves logistic_model.pkl and cv.pkl
python test.py "Hello, how are you?" "URGENT! Free ticket!!!"
```

## Usage

1. Clone or download this repository.
2. Ensure you have Python 3 with `pandas`, `numpy`, `scikit‑learn` and optionally `nltk` installed.
3. Run `python train.py` to train the model.
4. Run `python test.py` with one or more messages to classify them.

## Notes

* The logistic regression model provides a simple baseline; feel free to experiment with other algorithms or preprocessing techniques.
* The original Jupyter notebooks are retained for reference but rely on TensorFlow/Keras and may not run without those packages installed.

# SMS Spam Detection

This project contains a simple machine‑learning model that classifies SMS messages as **spam** or **ham** (not spam).  It was built as a proof‑of‑concept for filtering unsolicited messages.

## Dataset

The model is trained on the widely used SMS Spam Collection dataset, which contains 5,572 English SMS messages labeled as `ham` (legitimate) or `spam`.  A copy of the dataset is included in **`spam.csv`** (columns beyond the first two are removed).

## Training process

The training process is implemented in the notebook **`SMS Spam Detection train notebook.ipynb`** and involves the following steps:

1. **Load and clean the data** – The messages are loaded from `spam.csv`.  Extra columns are dropped, and each message is converted to lowercase with punctuation and numbers removed.

2. **Vectorize** – A `CountVectorizer` from scikit‑learn converts the cleaned text to a bag‑of‑words representation using the 100 most frequent tokens.  The fitted vectorizer is saved to **`cv.pkl`** so it can be reused for inference.

3. **Model architecture** – A simple feed‑forward neural network is built using Keras (TensorFlow backend).  The network has:
   * An input layer of size 100 (the number of features from the vectorizer).
   * Two hidden layers with 8 neurons each and ReLU activations.
   * An output layer with a single neuron and sigmoid activation for binary classification.

4. **Training** – The model is compiled with the Adam optimizer and binary cross‑entropy loss.  It is trained for 200 epochs on 80 % of the data with a batch size of 4,048.  After training, the model is saved as **`mymodel.h5`**.

## Testing / inference

The notebook **`sms spam detection test notebook .ipynb`** demonstrates how to load the trained model and vectorizer and use them to classify new messages.  The key steps are:

```python
from keras.models import load_model
import pickle

# Load vectorizer and model
cv = pickle.load(open('cv.pkl', 'rb'))
model = load_model('mymodel.h5')

# Transform a new message and predict
new_message = "URGENT! You have won a free ticket!"
X = cv.transform([new_message]).toarray()
prediction = model.predict(X)
label = "spam" if prediction > 0.5 else "ham"
print(label)
```

## Usage

1. Clone or download this repository.
2. Install the necessary Python packages (e.g. `pandas`, `numpy`, `scikit‑learn`, `tensorflow`, and `nltk`).

```bash
pip install pandas numpy scikit-learn tensorflow nltk
```

3. Open the training notebook (`SMS Spam Detection train notebook.ipynb`) in Jupyter to reproduce training (optional), or the test notebook (`sms spam detection test notebook .ipynb`) to classify your own messages.

4. Use the files **`cv.pkl`** and **`mymodel.h5`** in your own scripts to perform inference without the notebooks.

## Notes

* The model provided here is a basic proof of concept and is not optimized for production use.
* Accuracy will depend on pre‑processing choices; feel free to experiment with stemming, stop‑word removal or different model architectures to improve performance.
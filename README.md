# Movie Review Sentiment Analysis using Simple RNN

A project that uses a Simple Recurrent Neural Network to perform sentiment analysis on movie reviews (positive vs negative). Built in Python & Jupyter notebooks. Includes training, prediction, and embedding visualisation.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Setup & Installation](#setup--installation)
4. [Usage](#usage)

   * Training the model
   * Making predictions
   * Embedding visualization
5. [Files & Notebooks Explained](#files--notebooks-explained)
6. [Dependencies](#dependencies)
7. [Model Details](#model-details)
8. [Results](#results)
9. [Future Improvements](#future-improvements)
10. [License](#license)
11. [Contact](#contact)

---

## 1. Project Overview

This project aims to classify movie reviews (from e.g. the IMDB dataset) into positive or negative sentiment using a **Simple RNN** (Recurrent Neural Network). Key goals:

* Preprocess text data (tokenization, padding, embedding)
* Build and train an RNN model
* Evaluate performance on test data
* Provide interface to make predictions on new reviews
* Visualize embeddings to gain insight into how words are encoded

---

## 2. Repository Structure

Here’s what the repository contains:

| File / Folder        | Description                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------------- |
| `simplernn.ipynb`    | Notebook where the RNN model is built, trained, and evaluated.                                |
| `embedding.ipynb`    | Notebook for learning / extracting word embeddings, possibly visualizing them.                |
| `prediction.ipynb`   | Notebook for loading the trained model and using it to predict sentiment on new text/reviews. |
| `main.py`            | Python script (non-notebook) version to run training / predictions (if applicable).           |
| `requirements.txt`   | List of Python package dependencies.                                                          |
| `simple_rnn_imdb.h5` | Trained RNN model file (saved weights + architecture) in H5 format.                           |
| `.gitignore`         | File to ignore temporary / unnecessary files.                                                 |

---

## 3. Setup & Installation

To run this project on your local machine, follow these steps:

1. **Clone the repository**

   ```bash
   git clone https://github.com/Ranjan83711/Movie-Review---Using-Simple-RNN.git
   cd Movie-Review---Using-Simple-RNN
   ```

2. **Create a new virtual environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # on Linux/macOS
   venv\Scripts\activate      # on Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify setup**

   * Python version should be compatible (e.g. Python 3.6 or above)
   * Jupyter installed (if you want to run notebooks)
   * Enough memory so training RNN is feasible

---

## 4. Usage

### Training the model

* Open `simplernn.ipynb` and run all cells. This notebook includes loading data, preprocessing, defining the RNN architecture, training, evaluating, and saving the model to `simple_rnn_imdb.h5`.
* Alternatively, if `main.py` is set up to run training, you can do:

  ```bash
  python main.py --train
  ```

  *(Assumes `main.py` supports a train mode — you may need to read/adjust its code.)*

### Making predictions

* Use `prediction.ipynb` to load the saved model (`simple_rnn_imdb.h5`) and test it on new review texts.
* In code, something like:

  ```python
  from tensorflow.keras.models import load_model
  model = load_model("simple_rnn_imdb.h5")
  # preprocess your text (tokenize, pad) then model.predict(...)
  ```

### Embedding visualization

* Use `embedding.ipynb` to explore the word embeddings learned by the model. Visualize them (e.g., via TSNE or PCA) to understand which words are clustered.
* Useful for seeing how positive vs negative words are represented.

---

## 5. Files & Notebooks Explained

* **simplernn.ipynb** — the core notebook: data preprocessing, model building, training & evaluation.
* **embedding.ipynb** — focuses on word embedding layers: extracting embedding weights, visualizing with dimensionality reduction.
* **prediction.ipynb** — for inference: loading model and using on new/unseen data.
* **main.py** — a script version (if present) to do training / prediction without notebook.
* **requirements.txt** — all packages needed (TensorFlow / Keras, numpy, pandas, etc.).
* **simple\_rnn\_imdb.h5** — saved model (architecture + weights). Use this for predictions without retraining.

---

## 6. Dependencies

Typical dependencies include:

* Python 3.x
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn (for metrics, maybe for TSNE or PCA)
* Matplotlib / Seaborn (for visualization)
* Jupyter notebook

Check `requirements.txt` for exact versions.

---

## 7. Model Details

* **Model type**: Simple RNN (Recurrent Neural Network)
* **Input preprocessing**: tokenization of movie reviews, converting to sequences, padding/truncating to fixed length.
* **Embedding layer**: learnable embedding (word vectors) as first layer.
* **RNN layer(s)**: SimpleRNN units; possibly with dropout, etc.
* **Output**: binary classification (positive / negative), likely a sigmoid output.
* **Loss function**: binary crossentropy
* **Optimizer**: e.g. Adam or similar
* **Metrics**: accuracy, possibly precision/recall, etc.

---

## 8. Results

* Include (or you may fill in) metrics such as training accuracy/loss, validation accuracy/loss.
* Maybe include confusion matrix, ROC-AUC, etc.
* If embedding visualizations are available: show some clusters of words, maybe interesting observations (e.g. “good”, “excellent” cluster separate from “bad”, “awful”, etc.).

---

## 9. Future Improvements

Some ideas to extend or improve the project:

* Replace SimpleRNN with **LSTM** or **GRU** to see if performance improves.
* Use **bidirectional RNN**.
* Add regularization (dropout, recurrent dropout).
* Use pre-trained word embeddings (e.g. GloVe, Word2Vec) rather than learning from scratch.
* Handle more nuanced sentiment (e.g. multi-class: very negative, negative, neutral, positive, very positive).
* Improve text preprocessing (remove stop words, lemmatization, etc.).
* Hyperparameter tuning (number of layers, units, learning rate, sequence length).
* Deployment: build flask / fastAPI app to expose predictions via API.

---



If you want, I can generate a ready-made Markdown README file (with badges etc.) for this repo — do you want me to prepare that for you?

<div align="center">
  <h1>IMDb Movie Review Sentiment Analysis</h1>
  <p><strong>A Simple RNN that predicts whether a movie review expresses positive or negative sentiment.</strong></p>
  <p>
    <a href="https://imdb-movie-review-rnn-a6jnjisnsc6oappowjlnurg.streamlit.app/">Open the live app</a>
  </p>
</div>

<p align="center">
  <img src="photo/IMDb%20Movie%20Sentiment%20Analysis%20Banner.png" alt="IMDb Movie Review Sentiment Analysis banner" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow&logoColor=white" alt="TensorFlow 2.15">
  <img src="https://img.shields.io/badge/Streamlit-app-red?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-GPL--3.0-green" alt="GPL-3.0 license">
</p>

## Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Features](#features)
- [How it works](#how-it-works)
- [Model and results](#model)
- [Dataset](#dataset)
- [Repository contents](#repository-contents)
- [Run locally](#run-locally)
- [Limitations](#limitations)
- [Possible next steps](#possible-next-steps)
- [License](#license)

## Overview

This project is a small end-to-end natural language processing application. It uses a trained TensorFlow/Keras Simple Recurrent Neural Network (RNN) to classify a written movie review as **Positive** or **Negative**. The Streamlit interface accepts a review, displays its word and character counts, and presents the predicted sentiment, model score, and confidence-style score.

## Live Demo

[Try the IMDb Sentiment Analyzer](https://imdb-movie-review-rnn-a6jnjisnsc6oappowjlnurg.streamlit.app/)

### App screenshot

![Screenshot of the IMDb Sentiment Analyzer Streamlit app](<photo/Screenshot 2026-11-25 110737.png>)

## Features

- Classifies free-form movie reviews into positive or negative sentiment.
- Shows the raw sigmoid prediction score and a confidence-style percentage.
- Displays review word and character counts.
- Uses the same IMDb word-index mapping and sequence length as the trained model.
- Includes notebooks for model training, prediction experiments, and embedding exploration.

## Tech stack

| Area | Technology |
|---|---|
| Application | Streamlit |
| Language | Python |
| Deep learning | TensorFlow / Keras |
| Dataset and word index | Keras IMDb dataset |

## How it works

```text
Movie review
    -> lowercase and tokenize
    -> map words to IMDb vocabulary IDs
    -> pad or truncate to 500 tokens
    -> Embedding layer
    -> SimpleRNN layer
    -> sigmoid score
    -> Positive or Negative label
```

The app tokenizes input into lowercase words, maps known words using the IMDb word index, and maps out-of-vocabulary words to an unknown-token ID. It prepares a sequence of 500 tokens before passing it to the model. A score greater than `0.5` is labeled **Positive**; otherwise, the result is **Negative**. The displayed confidence-style score is the model score for the selected class; it is not a separately calibrated probability.

## Model

| Component | Configuration |
|---|---|
| Dataset vocabulary | Up to 10,000 words |
| Input sequence length | 500 tokens |
| Embedding | 150 dimensions |
| Recurrent layer | SimpleRNN, 128 units, `tanh` activation |
| Output | One unit with `sigmoid` activation |
| Training objective | Binary cross-entropy |
| Optimizer | Adam |

The training notebook uses batches of 32, a 20% validation split, and up to 10 epochs with early stopping on validation accuracy. The saved model used by the app is `artifacts/simple_rnn_imdb.h5`.

### Recorded training result

The run recorded in `simplernn.ipynb` reached a best validation accuracy of **79.28%** (epoch 4). This is a validation result, **not a held-out test score**; the notebook does not report a separate test-set evaluation metric.

## Dataset

The notebooks use the IMDb movie-review dataset provided by `tensorflow.keras.datasets.imdb`. It contains 50,000 labeled reviews, with 25,000 examples for training and 25,000 for testing. The training workflow limits the vocabulary to 10,000 words and pads sequences to 500 tokens.

The raw dataset is fetched by Keras when needed rather than stored in this repository. On first startup, the app may also need an internet connection to retrieve the IMDb word index if it is not already cached.

## Repository contents

```text
IMDb-Movie-Review-RNN/
├── artifacts/
│   └── simple_rnn_imdb.h5   # Trained model loaded by the app
├── photo/
│   ├── IMDb Movie Sentiment Analysis Banner.png
│   └── Screenshot 2026-11-25 110737.png
├── embedding.ipynb          # Embedding experiment
├── prediction.ipynb         # Prediction and inference experiment
├── simplernn.ipynb           # Model training notebook
├── main.py                   # Streamlit application
├── requirements.txt          # Python dependencies
├── LICENSE                   # GNU General Public License v3.0
└── README.md
```

## Run locally

Use Python 3.11 and run these commands from the project directory.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run main.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run main.py
```

The trained model must remain at `artifacts/simple_rnn_imdb.h5`, relative to the project root. Streamlit will print a local URL after startup; open it in a browser to use the app.

## Limitations

- This is a binary sentiment classifier; it does not explain why a prediction was made.
- Sarcasm, mixed opinions, unusual phrasing, and words outside the model's vocabulary can lead to incorrect predictions.
- Reviews are limited to 500 tokens, so longer text is truncated.
- The displayed confidence-style value is derived directly from the sigmoid score and has not been calibrated.
- The recorded validation result should not be treated as a measure of test-set or real-world performance.

## Possible next steps

- Evaluate the saved model on the held-out IMDb test set and report the results.
- Compare the Simple RNN with LSTM/GRU or transformer-based models.
- Add model explanations and confidence calibration.
- Add automated tests for preprocessing and prediction behavior.

## License

This project is distributed under the **GNU General Public License v3.0**. See [LICENSE](LICENSE) for the complete terms.

## Author

**Samsure Rahman**

## Acknowledgements

- IMDb review dataset distributed through [Keras](https://keras.io/api/datasets/imdb/).
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for model training and inference.
- [Streamlit](https://streamlit.io/) for the interactive web application.

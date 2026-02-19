#Machine Learning & Data Mining

Sentiment classification on the **IMDB Movie Reviews** dataset using classical machine learning methods. This project implements and compares **Logistic Regression** and **k-Nearest Neighbors (k-NN)** classifiers with TF-IDF text vectorization.

## Overview

The goal is to classify movie reviews as positive or negative based on their text. The dataset contains 25,000 training and 25,000 test reviews from the IMDB dataset.

### Models

| Model | Description |
|-------|-------------|
| **Logistic Regression** | L2-regularized logistic regression with TF-IDF features |
| **k-NN** | K-Nearest Neighbors classifier with hyperparameter search over k and training set size |

## Project Structure

```
.
├── data.py              # Dataset loading via TensorFlow Datasets (with optional caching)
├── logistic.py          # Logistic regression sentiment classifier
├── knn_classifier.ipynb # k-NN experiments and visualizations
├── requirements.txt     # Python dependencies
└── README.md
```

## Setup

### Prerequisites

- Python 3.11 (see `.python-version`)
- [Pyenv](https://github.com/pyenv/pyenv) or similar (optional, for version management)

### Installation

1. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **First run**: The IMDB dataset will be downloaded automatically via TensorFlow Datasets. Use `cache=True` in `load_dataset()` to pickle the data for faster subsequent loads.

## Usage

### Logistic Regression

Run the logistic regression classifier (uses `data.py` and TensorFlow Datasets):

```bash
python logistic.py
```

Output includes train and test accuracy. Typical results are in the ~85–90% range.

### k-NN Classifier

Open and run the Jupyter notebook:

```bash
jupyter notebook knn_classifier.ipynb
# or
jupyter lab knn_classifier.ipynb
```

The notebook explores:

- **Sample sizes**: 1,000 / 5,000 / 10,000 / 20,000 training examples
- **k values**: 1, 3, 5, 7, 9
- Train vs. test accuracy and learning curves

> **Note:** The notebook uses `torchtext.datasets.IMDB` for data loading. If you prefer the same data source as `logistic.py`, you can adapt it to use `data.load_dataset()` from this project instead.

## Data Module (`data.py`)

- **`load_dataset(name, cache_dir="cache", cache=False)`**  
  Loads the IMDB dataset via TensorFlow Datasets. With `cache=True`, data is pickled in `cache/` for faster reloading.
- **`tfds_as_np(ds)`**  
  Converts a TFDS dataset to a NumPy array of `(text, label)` tuples.

## Results (from k-NN notebook)

From the notebook experiments:

- **Best k** (with 20k samples): k = 9 (~66.3% test accuracy)
- **Observation**: Higher k generally performs better on this dataset; train–test gap indicates overfitting at small sample sizes.

## Dependencies

Key packages:

- `tensorflow` & `tensorflow-datasets` — dataset loading
- `scikit-learn` — LogisticRegression, KNeighborsClassifier, TfidfVectorizer, metrics
- `numpy` — array operations
- `jupyter`, `matplotlib`, `pandas` — notebook and visualization

See `requirements.txt` for full versions.

## License

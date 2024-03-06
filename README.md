# Cosine Similarity Search with TF-IDF and FastText

This project provides two Python classes, `CosineTFIDFFinder` and `CosineFastTextFinder`, for searching text datasets. `CosineTFIDFFinder` utilizes TF-IDF vectors, while `CosineFastTextFinder` uses FastText embeddings. Both leverage cosine similarity to identify the most similar items in a dataset based on a query text.

## Requirements and Installation

Before you start, ensure you have Python 3.6 or later installed on your system. You will need the following libraries:

- numpy
- scikit-learn
- fasttext (only for `CosineFastTextFinder`)

To install the dependencies, run the following command:

```bash
pip install numpy scikit-learn fasttext
```

## Quickstart

Using CosineTFIDFFinder

```bash
from CosineTFIDFFinder import CosineTFIDFFinder

# Initialize finder and preprocess dataset
finder = CosineTFIDFFinder()
finder.preprocess_dataset(['your', 'dataset', 'texts'], lambda x: x)

# Search for similar texts
results = finder.search('query text')
```

Using CosineFastTextFinder

```bash
from CosineFastTextFinder import CosineFastTextFinder

# Initialize finder with path to FastText model
finder = CosineFastTextFinder('path_to_fasttext_model.bin')

# Preprocess dataset
finder.preprocess_dataset(['your', 'dataset', 'texts'], lambda x: x)

# Search for similar texts
results = finder.search('query text')
```

## API Reference
CosineTFIDFFinder
```bash
preprocess_dataset(dataset, transform): Prepares the dataset by transforming it into TF-IDF vectors.

search(query, similarity_threshold=0.5, top_n=5): Searches for the top N most similar texts in the dataset to the query.
```
CosineFastTextFinder
```bash
preprocess_dataset(dataset, transform): Transforms and vectorizes the dataset texts using the FastText model.

search(query, similarity_threshold=0.5, top_n=5): Searches for the most similar dataset
texts to the query.
```

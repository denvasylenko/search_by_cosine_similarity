from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class CosinceTFIDFFinder:
    """
    A class for performing TF-IDF vectorization and cosine similarity based search on a text dataset.

    This class allows for preprocessing a dataset to transform it into TF-IDF vector form, and
    then querying this dataset to find items most similar to a given text input based on cosine similarity.

    Attributes:
        tfidf_vectorizer (TfidfVectorizer): The vectorizer used to convert texts to TF-IDF vectors.
        dataset_vectors (numpy.ndarray): The TF-IDF vectors for the preprocessed dataset.
    """

    def __init__(self):
        """
        Initializes the CosinceTFIDFFinder class without any dataset. Before performing search,
        a dataset must be preprocessed using the preprocess_dataset method.
        """
        self.tfidf_vectorizer = None
        self.dataset_vectors = None

    def preprocess_dataset(self, dataset, transform):
        """
        Preprocesses the dataset by applying a transformation function to each item and then
        converting the transformed dataset to TF-IDF vectors.

        Parameters:
            dataset (List[Any]): The dataset to be processed. It can contain any type of elements,
                                 as long as the transform function can handle them.
            transform (Callable[[Any], str]): A function that takes an item from the dataset as input
                                              and returns a string representation of it.
        """
        self.dataset = dataset
        datasetText = [transform(item) for item in self.dataset]

        # Initialize a TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer()
        # Transform the dataset into TF-IDF vectors
        self.dataset_vectors = self.tfidf_vectorizer.fit_transform(datasetText).toarray()

    def text_to_vector(self, text):
        """
        Converts a single text input to a TF-IDF vector using the previously trained TF-IDF model.

        Parameters:
            text (str): The text input to convert.

        Returns:
            numpy.ndarray: The TF-IDF vector representation of the input text.
        """
        return self.tfidf_vectorizer.transform([text]).toarray()

    def search(self, query, similarity_threshold=0.5, top_n=5):
        """
        Searches the dataset for the top_n items most similar to the query text, based on cosine similarity
        of their TF-IDF vector representations.

        Parameters:
            query (str): The query text.
            similarity_threshold (float): The minimum similarity score that an item must have to the
                                          query to be considered as a result.
            top_n (int): The maximum number of most similar items to return.

        Returns:
            List[Any]: A list of the top_n most similar items in the dataset to the query text.
                       If fewer than top_n items meet the similarity threshold, a shorter list will be returned.
        """
        query_vector = self.text_to_vector(query)
        cos_similarities = cosine_similarity(query_vector, self.dataset_vectors).squeeze(0)

        eligible_indices = np.where(cos_similarities >= similarity_threshold)[0]
        if len(eligible_indices) > 0:
            top_n_indices = np.argsort(cos_similarities[eligible_indices])[-top_n:][::-1]
            top_items_indices = eligible_indices[top_n_indices]
            top_items = [self.dataset[idx] for idx in top_items_indices]
        else:
            top_items = []

        return top_items

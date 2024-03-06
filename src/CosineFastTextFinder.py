import numpy as np
import fasttext
from sklearn.metrics.pairwise import cosine_similarity

class CosineFastTextFinder:
    """
    A class to perform efficient text search using FastText word embeddings and cosine similarity.

    This class enables the pre-processing of a text dataset into vector representations using FastText,
    and provides functionality to search for the most similar items in the dataset based on cosine similarity.

    Attributes:
        ft_model (fasttext.FastText._FastText): The FastText model loaded from a specified path.
        dataset (List[str]): A list of text items constituting the dataset.
        dataset_vectors (numpy.ndarray): A NumPy array of vector representations of the dataset items.

    Methods:
        __init__(model_path): Initializes the CosineFastTextFinder instance by loading a FastText model.
        preprocess_dataset(dataset, transform): Pre-processes the dataset, applying a transformation function and converting text to vectors.
        text_to_vector(text): Converts a single text item to its vector representation using the FastText model.
        calc_cos_similarities(query_vector): Calculates the cosine similarities between a query vector and all dataset vectors.
        search(query, similarity_threshold, top_n): Searches for the top_n most similar items in the dataset to the query text.
    """

    def __init__(self, model_path):
        """
        Initializes the CosineFastTextFinder instance.

        Parameters:
            model_path (str): The path to the FastText model file.
        """
        self.ft_model = fasttext.load_model(model_path)
        self.dataset = []
        self.dataset_vectors = np.array([])

    def preprocess_dataset(self, dataset, transform):
        """
        Pre-processes the dataset by applying a transformation function to each item, converting them to strings if necessary, and then converting those strings to vectors.

        This method allows the dataset to consist of any type of items, not limited to strings. The transformation function provided should be capable of converting these items into a string format that can be processed by the text_to_vector method.

        Parameters:
            dataset (List[Any]): The list of items to be processed. Items in this list can be of any type.
            transform (Callable[[Any], str]): A function to apply to each item in the dataset. This function should take a single item of any type as input and return a string representation of that item suitable for text processing and vectorization.

        The transformed dataset is then converted into vector representations using the FastText model.
        """
        self.dataset = dataset
        data_text_set = [transform(item) for item in self.dataset]
        self.dataset_vectors = np.array([self.text_to_vector(text) for text in data_text_set])

    def text_to_vector(self, text):
        """
        Converts a single text item to its vector representation using the FastText model.

        Parameters:
            text (str): The text item to convert.

        Returns:
            numpy.ndarray: The vector representation of the input text.
        """
        words = fasttext.tokenize(text)
        vectors = np.array([self.ft_model.get_word_vector(word) for word in words if word])

        if vectors.size > 0:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.ft_model.get_dimension())

    def calc_cos_similarities(self, query_vector):
        """
        Calculates the cosine similarities between a query vector and all dataset vectors.

        Parameters:
            query_vector (numpy.ndarray): The query vector.

        Returns:
            numpy.ndarray: An array of cosine similarity scores.
        """
        query_vector_reshaped = query_vector.reshape(1, -1)
        cos_similarities = cosine_similarity(query_vector_reshaped, self.dataset_vectors)
        return cos_similarities.squeeze()

    def search(self, query, similarity_threshold=0.5, top_n=5):
        """
        Searches for the top_n most similar items in the dataset to the query text, based on cosine similarity. 

        This method leverages the vector representations of the dataset items and the query to compute cosine similarities, returning the items that are most similar to the query. The items returned are of the same type as those stored in the dataset, allowing for flexibility in the types of data that can be searched.

        Parameters:
            query (str): The query text, which will be converted to a vector representation for similarity comparison.
            similarity_threshold (float): The minimum similarity score that an item must have to the query vector to be considered as a result.
            top_n (int): The number of most similar items to return. If fewer than top_n items meet the similarity threshold, fewer items will be returned.

        Returns:
            List[Any]: A list of the top_n most similar items in the dataset. The type of items in this list matches the type of items in the dataset provided during preprocessing.
        """
        query_vector = self.text_to_vector(query)
        cos_similarities = self.calc_cos_similarities(query_vector)

        indices = np.where(cos_similarities >= similarity_threshold)[0]
        if indices.size > 0:
            filtered_similarities = cos_similarities[indices]
            top_n_indices_desc = np.argsort(-filtered_similarities)[:top_n]
            top_items_indices = indices[top_n_indices_desc]
            top_items = [self.dataset[idx] for idx in top_items_indices]
        else:
            top_items = []

        return top_items

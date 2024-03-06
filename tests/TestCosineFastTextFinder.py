import unittest
import fasttext
from unittest.mock import MagicMock
import numpy as np
from CosineFastTextFinder import CosineFastTextFinder

class TestCosineFastTextFinder(unittest.TestCase):
    def setUp(self):
        # Mock the FastText model to return predefined vectors for specific words
        self.mock_model = MagicMock()
        self.mock_model.get_word_vector.return_value = np.array([1.0, 2.0, 3.0])
        self.mock_model.get_dimension.return_value = 3

        # Patch the fasttext.load_model to return the mock model
        fasttext.load_model = MagicMock(return_value=self.mock_model)

        # Initialize the CosineFastTextFinder instance with a mock model path
        self.searcher = CosineFastTextFinder(model_path="mock/model/path")

    def test_text_to_vector(self):
        # Test that text_to_vector returns the expected average vector
        result = self.searcher.text_to_vector("word1 word2")
        expected = np.array([1.0, 2.0, 3.0])  # Expected vector based on the mock model's return value
        np.testing.assert_array_almost_equal(result, expected)

    def test_preprocess_dataset(self):
        # Test that preprocessing a dataset stores the correct vectors
        self.searcher.preprocess_dataset(["word1", "word2"], lambda x: x)
        expected_vectors = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        np.testing.assert_array_almost_equal(self.searcher.dataset_vectors, expected_vectors)

    def test_calc_cos_similarities(self):
        # Preprocess a mock dataset
        self.searcher.preprocess_dataset(["word1", "word2"], lambda x: x)
        query_vector = np.array([1.0, 2.0, 3.0])
        similarities = self.searcher.calc_cos_similarities(query_vector)
        # Expect similarities to be a 1D array with values close to 1 (high similarity)
        self.assertEqual(similarities.shape, (2,))
        np.testing.assert_array_almost_equal(similarities, np.array([1.0, 1.0]))

    def test_search(self):
        # Preprocess a mock dataset
        self.searcher.preprocess_dataset(["word1", "word2"], lambda x: x)
        # Search for a query that matches the mock dataset items
        results = self.searcher.search("word1", similarity_threshold=0.99, top_n=1)
        # Expect the search to return the first item
        self.assertEqual(results, ["word1"])

if __name__ == '__main__':
    unittest.main()

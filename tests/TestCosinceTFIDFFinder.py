import unittest
from CosinceTFIDFFinder import TFIDFSearch
from sklearn.feature_extraction.text import TfidfVectorizer

class TestCosinceTFIDFFinder(unittest.TestCase):
    def setUp(self):
        """Set up a TFIDFSearch object and a sample dataset for testing."""
        self.searcher = TFIDFSearch()
        self.dataset = [
            "The quick brown fox jumps over the lazy dog",
            "Never jump over the lazy dog quickly",
            "Bright vixens jump; dozy fowl quack",
            "Quick wafting zephyrs vex bold Jim"
        ]
        # Simple transform function that returns the text itself for this example
        self.transform = lambda x: x
    
    def test_preprocess_dataset(self):
        """Test the preprocess_dataset method with a more robust approach."""
        self.searcher.preprocess_dataset(self.dataset, self.transform)
        # Ensure dataset_vectors is initialized
        self.assertIsNotNone(self.searcher.dataset_vectors)
        # Directly calculate the expected shape using a similar TF-IDF process
        test_vectorizer = TfidfVectorizer()
        test_vectors = test_vectorizer.fit_transform([self.transform(item) for item in self.dataset]).toarray()
        # Compare shapes
        self.assertEqual(self.searcher.dataset_vectors.shape, test_vectors.shape)

    
    def test_text_to_vector(self):
        """Test the text_to_vector method."""
        self.searcher.preprocess_dataset(self.dataset, self.transform)
        vector = self.searcher.text_to_vector("quick fox")
        # Check if the vector is not None and has the correct shape
        self.assertIsNotNone(vector)
        self.assertEqual(vector.shape, (1, self.searcher.dataset_vectors.shape[1]))
    
    def test_search(self):
        """Test the search method."""
        self.searcher.preprocess_dataset(self.dataset, self.transform)
        results = self.searcher.search("quick fox", similarity_threshold=0.1, top_n=2)
        # Check if results is not empty and has at most 2 items
        self.assertIsNotNone(results)
        self.assertTrue(len(results) <= 2)
        # Check if the first item is the most relevant to the query "quick fox"
        self.assertIn("The quick brown fox jumps over the lazy dog", results)

if __name__ == '__main__':
    unittest.main()

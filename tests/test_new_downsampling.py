import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import MagicMock

# Mock vertexai before importing utils/downsampling
sys.modules['vertexai'] = MagicMock()
sys.modules['vertexai.language_models'] = MagicMock()
sys.modules['networkx'] = MagicMock()

# Mock sklearn
mock_sklearn = MagicMock()
sys.modules['sklearn'] = mock_sklearn
sys.modules['sklearn.metrics'] = MagicMock()
sys.modules['sklearn.metrics.pairwise'] = MagicMock()
sys.modules['sklearn.cluster'] = MagicMock()

# Define side effects for mocks
def mock_cosine_similarity(X, Y=None):
    if Y is None:
        Y = X
    # Return random similarity matrix
    return np.random.rand(len(X), len(Y))

def mock_pairwise_distances_argmin_min(X, Y):
    # Return random indices and distances
    # X is centroids (k), Y is data (N)
    # We want for each centroid, the closest data point index.
    # Returns (indices, distances)
    # indices shape (k,)
    return np.random.randint(0, len(Y), size=len(X)), np.random.rand(len(X))

sys.modules['sklearn.metrics.pairwise'].cosine_similarity = mock_cosine_similarity
sys.modules['sklearn.metrics.pairwise'].pairwise_distances_argmin_min = mock_pairwise_distances_argmin_min

# Mock KMeans
class MockKMeans:
    def __init__(self, n_clusters=8, **kwargs):
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        self.cluster_centers_ = np.random.rand(self.n_clusters, X.shape[1])
        return self

    def fit_predict(self, X):
        self.fit(X)
        self.labels_ = np.random.randint(0, self.n_clusters, size=len(X))
        return self.labels_

sys.modules['sklearn.cluster'].KMeans = MockKMeans

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from downsampling import sample_kmeans, sample_dedup, apply_downsampling

class TestDownsampling(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'text': [f'Sentence {i}' for i in range(20)],
            'label': [0] * 10 + [1] * 10
        })
        self.k = 5

    @patch('downsampling.get_embeddings_task')
    def test_sample_kmeans(self, mock_get_embeddings):
        # Mock embeddings: 20 samples, 10 dimensions
        mock_get_embeddings.return_value = np.random.rand(20, 10)
        
        result = sample_kmeans(self.data, k_samples=self.k)
        
        self.assertEqual(len(result), self.k)
        self.assertTrue(all(col in result.columns for col in self.data.columns))

    @patch('downsampling.get_embeddings_task')
    def test_sample_dedup(self, mock_get_embeddings):
        # Mock embeddings
        mock_get_embeddings.return_value = np.random.rand(20, 10)
        
        result = sample_dedup(self.data, k_samples=self.k)
        
        self.assertEqual(len(result), self.k)
        self.assertTrue(all(col in result.columns for col in self.data.columns))

    @patch('downsampling.get_embeddings_task')
    def test_sample_dedup_high_similarity(self, mock_get_embeddings):
        # Create embeddings where some are identical to force pruning
        embeddings = np.random.rand(20, 10)
        embeddings[1] = embeddings[0] # Duplicate
        embeddings[3] = embeddings[2] # Duplicate
        mock_get_embeddings.return_value = embeddings
        
        # We set k=18, so duplicates should be removed but we might need to fill up?
        # Actually dedup logic:
        # 1. Cluster (k=18) -> almost one per cluster
        # 2. Prune duplicates within cluster.
        # If we set k small, say k=5, clustering will group them.
        
        result = sample_dedup(self.data, k_samples=5, dedup_sim_threshold=0.99)
        self.assertEqual(len(result), 5)

    @patch('downsampling.get_embeddings_task')
    def test_apply_downsampling_integration(self, mock_get_embeddings):
        mock_get_embeddings.return_value = np.random.rand(20, 10)
        
        # Test via dispatcher
        res_kmeans = apply_downsampling(self.data, method='kmeans', target_size=5)
        self.assertEqual(len(res_kmeans), 5)
        
        res_dedup = apply_downsampling(self.data, method='dedup', target_size=5)
        self.assertEqual(len(res_dedup), 5)

if __name__ == '__main__':
    unittest.main()

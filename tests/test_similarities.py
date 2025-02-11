import pytest
import numpy as np
import sqlite3
from unittest.mock import patch, MagicMock
import sys
import os

# weil sich similarities.py nicht im selben ordner befindet (sieht so cleaner aus :D)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from similarities import calculate_similarities, load_feature_data
from database import select_image_from_database
from histograms import input_image_histogram
from hashes import input_image_ahash, input_image_dhash
from embeddings import input_image_embedding

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from similarities import calculate_similarities

# Mock-Daten für das Feature-Set
MOCK_FEATURE_DATA = [
    {"image_id": 1, "rgb_hists": np.random.rand(256), "embeddings": np.random.rand(128), "ahashes": np.random.rand(64), "dhashes": np.random.rand(64)},
    {"image_id": 2, "rgb_hists": np.random.rand(256), "embeddings": np.random.rand(128), "ahashes": np.random.rand(64), "dhashes": np.random.rand(64)},
]

@pytest.fixture
def mock_cursor():
    conn = MagicMock()
    cursor = conn.cursor()
    yield cursor

def mock_histogram(image_path):
    return np.random.rand(256)

def mock_embedding(image_path):
    return np.random.rand(128)

@patch("similarities.load_feature_data", return_value=MOCK_FEATURE_DATA)
@patch("database.select_image_from_database", return_value="mock_path/image_1.jpg")
@patch("histograms.input_image_histogram", side_effect=mock_histogram)
@patch("embeddings.input_image_embedding", side_effect=mock_embedding)
def test_calculate_similarities_single_case(mock_load_data, mock_select_image, mock_hist, mock_emb, mock_cursor):
    input_images = ["mock_path/image_1.jpg", "mock_path/image_2.jpg"]

    with patch("PIL.Image.open"):
        try:
            calculate_similarities(input_images, mock_cursor, "rgb", "cosine", top_k=2, verbose=False)
        except Exception as e:
            pytest.fail(f"calculate_similarities failed with error: {e}")
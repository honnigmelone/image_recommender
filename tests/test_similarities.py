import pytest
import numpy as np
import sqlite3
from unittest.mock import patch
import sys

# weil sich similarities.py nicht im selben ordner befindet (sieht so cleaner aus :D)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from similarities import calculate_similarities, load_feature_data
from database import select_image_from_database


# beispielhafte fake-daten für das extrahierte feature-Set
MOCK_FEATURE_DATA = [
    {"image_id": 1, "rgb_hists": np.random.rand(256), "embeddings": np.random.rand(128), "ahashes": np.random.rand(64), "dhashes": np.random.rand(64)},
    {"image_id": 2, "rgb_hists": np.random.rand(256), "embeddings": np.random.rand(128), "ahashes": np.random.rand(64), "dhashes": np.random.rand(64)},
]

# dummy-funktion für `select_image_from_database`, um datenbankaufrufe zu vermeiden
def mock_select_image_from_database(image_id, cursor):
    return f"mock_path/image_{image_id}.jpg"

@pytest.fixture
def mock_cursor():
    """
    mockt einen SQLite-cursor
    """
    # in memory datenbank, also keine richtige datenbank 
    conn = sqlite3.connect(":memory:") 
    return conn.cursor()

@pytest.fixture
def mock_image():
    """
    mockt eine beispiel-input-image-datei
    """
    return ["test_image.jpg"]

@pytest.mark.parametrize("mode, metric", [
    ("rgb", "cosine"),
    ("rgb", "euclidean"),
    ("embeddings", "cosine"),
    ("embeddings", "euclidean"),
    ("ahashes", "hamming"),
    ("dhashes", "hamming"),
])
@patch("similarities.load_feature_data", return_value=MOCK_FEATURE_DATA)
@patch("database.select_image_from_database", side_effect=mock_select_image_from_database)
def test_calculate_similarities(mock_load_data, mock_select_image, mode, metric, mock_cursor, mock_image):
    """
    testet die funktion `calculate_similarities` für verschiedene modi und distanzmetriken
    """
    try:
        calculate_similarities(mock_image, mock_cursor, mode, metric, top_k=2, verbose=True)
    except Exception as e:
        pytest.fail(f"calculate_similarities failed for mode={mode}, metric={metric} with error: {e}")
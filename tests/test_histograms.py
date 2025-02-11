import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from histograms import process_image, get_histogram, input_image_histogram


@pytest.fixture
def create_dummy_image(tmp_path):
    """Erstellt ein Dummy-Bild für die Tests."""
    # Erstelle ein Dummy-Bild mit zufälligen Farben
    image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(image_data)
    
    # Speichere das Dummy-Bild im temporären Testverzeichnis
    image_path = tmp_path / "dummy_image.jpg"
    image.save(image_path)
    
    return image_path


def test_process_image(create_dummy_image):
    """Testet die Verarbeitung eines Bildes."""
    image_path = create_dummy_image
    
    processed_image = process_image(image_path)
    
    assert isinstance(processed_image, Image.Image)
    assert processed_image.size == (100, 100)  # Überprüfe die Größe des Bildes
    assert processed_image.mode == "RGB"  # Überprüfe den Modus des Bildes


def test_get_histogram(create_dummy_image):
    """Testet die Berechnung des Farb-Histogramms."""
    image_path = create_dummy_image
    image = process_image(image_path)
    
    histogram = get_histogram(image)
    
    assert isinstance(histogram, np.ndarray)
    assert histogram.shape == (512,)  # Überprüfe die Form des Histogramms
    assert np.isclose(histogram.sum(), 1.0)  # Histogramm sollte normiert sein


def test_input_image_histogram(create_dummy_image):
    """Testet die Eingabebild-Histogrammfunktion."""
    image_path = create_dummy_image
    
    histogram = input_image_histogram(image_path)
    
    assert isinstance(histogram, np.ndarray)
    assert histogram.shape == (512,)  # Überprüfe die Form des Histogramms
    assert np.isclose(histogram.sum(), 1.0)  # Histogramm sollte normiert sein

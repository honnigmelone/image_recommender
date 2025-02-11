import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from hashes import process_image, get_ahash, get_dhash, input_image_ahash, input_image_dhash

@pytest.fixture
def create_dummy_image(tmp_path):
    """
    erstellt ein dummy bild für die tests
    """
    # dummy bild mit zufälligen farben
    image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(image_data)
    
    # speichert das dummy bild im temporären testverzeichnis
    image_path = tmp_path / "dummy_image.jpg"
    image.save(image_path)
    
    return image_path


def test_process_image(create_dummy_image):
    """
    testet die verarbeitung eines bildes
    """
    image_path = create_dummy_image
    
    processed_image = process_image(image_path)
    
    assert isinstance(processed_image, Image.Image)
    assert processed_image.size == (100, 100)  
    assert processed_image.mode == "RGB"  


def test_get_ahash(create_dummy_image):
    """
    testet die berechnung des perzeptuellen Hashes (hier: A-Hash)
    """
    image_path = create_dummy_image
    image = process_image(image_path)
    
    hash_array = get_ahash(image)
    
    assert isinstance(hash_array, np.ndarray)
    assert hash_array.shape == (32 * 32,)  # Überprüfe die Form des Hashes (1024,)
    assert np.all(np.isin(hash_array, [0, 1]))  # Hash sollte nur 0 und 1 enthalten


def test_get_dhash(create_dummy_image):
    """
    testet die berechnung des D-Hashes
    """
    image_path = create_dummy_image
    image = process_image(image_path)
    
    hash_array = get_dhash(image)
    
    assert isinstance(hash_array, np.ndarray)
    assert hash_array.shape == (32 * 32,)  
    assert np.all(np.isin(hash_array, [0, 1]))  

def test_input_image_ahash(create_dummy_image):
    """
    testet die eingabebild-a-hash-funktion
    """
    image_path = create_dummy_image
    
    hash_array = input_image_ahash(image_path)
    
    assert isinstance(hash_array, np.ndarray)
    assert hash_array.shape == (32 * 32,)  
    assert np.all(np.isin(hash_array, [0, 1]))  

def test_input_image_dhash(create_dummy_image):
    """
    testet die eingabebild-d-hash-funktion
    """
    image_path = create_dummy_image
    
    hash_array = input_image_dhash(image_path)
    
    assert isinstance(hash_array, np.ndarray)
    assert hash_array.shape == (32 * 32,)  
    assert np.all(np.isin(hash_array, [0, 1]))  
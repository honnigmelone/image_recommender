import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from embeddings import process_image, preprocess, get_embedding, input_image_embedding

# Dummy image path
dummy_image_path = "dummy_image.jpg"

@pytest.fixture
def mock_image_open():
    """Fixture to mock the opening of an image."""
    with patch("PIL.Image.open") as mock_open:
        mock_image = MagicMock(spec=Image.Image)
        mock_open.return_value = mock_image
        yield mock_open

@pytest.fixture
def mock_model():
    """Fixture to mock the ResNet model."""
    model = MagicMock()
    # Simulate model output as a random tensor
    model.return_value = torch.rand(1, 512)  # Assuming the embedding size is 512
    return model

def test_process_image(mock_image_open):
    """Test the process_image function."""
    processed_image = process_image(dummy_image_path)
    assert processed_image is not None, "The processed image should not be None."

def test_preprocess(mock_image_open):
    """Test the preprocess function."""
    mock_image = process_image(dummy_image_path)
    preprocessed_image = preprocess(mock_image)
    
    assert preprocessed_image.shape == (1, 3, 224, 224), "Preprocessed image should have shape (1, 3, 224, 224)."

@patch("embeddings.get_embedding", return_value=np.random.rand(512).astype(np.float32))
def test_get_embedding(mock_get_embedding, mock_image_open, mock_model):
    """Test the get_embedding function."""
    mock_image = process_image(dummy_image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    embedding = get_embedding(mock_image, mock_model, device)
    
    assert embedding is not None, "Embedding should not be None."
    assert len(embedding) == 512, "Embedding size should be 512."

@patch("embeddings.process_image", return_value=MagicMock(spec=Image.Image))
@patch("embeddings.get_embedding", return_value=np.random.rand(512).astype(np.float32))
def test_input_image_embedding(mock_process_image, mock_get_embedding):
    """Test the input_image_embedding function."""
    embedding = input_image_embedding(dummy_image_path)
    
    assert embedding is not None, "Embedding should not be None."
    assert len(embedding) == 512, "Embedding size should be 512."

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image
from io import BytesIO
import torch
import io
from torchvision import models
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from embeddings import process_image, preprocess, get_embedding, input_image_embedding


@pytest.fixture
def sample_image():
    """Erstellt ein einfaches 224x224 RGB-Testbild."""
    img = Image.new('RGB', (224, 224), color='blue')
    return img


def test_process_image(sample_image):
    """Testet, ob das Bild korrekt verarbeitet wird."""
    img_bytes = BytesIO()
    sample_image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    processed_img = process_image(img_bytes)
    assert processed_img.mode == "RGB"
    assert isinstance(processed_img, Image.Image)


def test_preprocess(sample_image):
    """Testet, ob die Preprocessing-Funktion ein Tensor-Format ausgibt."""
    tensor = preprocess(sample_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)  # Batch-Dimension und 3 Farbkanäle


def test_get_embedding(sample_image):
    """Testet die Embedding-Funktion mit ResNet18."""
    device = torch.device("cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    
    embedding = get_embedding(sample_image, model, device)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == 512  # ResNet18 gibt 512-dimensionale Vektoren aus


def test_input_image_embedding(sample_image, monkeypatch):
    """Testet die Funktion input_image_embedding mit einem RGB-Bild."""
    # Monkeypatching, um sicherzustellen, dass das Bild nicht erneut geöffnet wird
    monkeypatch.setattr("embeddings.process_image", lambda x: x)
    
    embedding = input_image_embedding(sample_image)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == 512  # ResNet18 gibt 512-dimensionale Vektoren aus



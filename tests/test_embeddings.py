import pytest
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import sys
import os
from torchvision import models
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from embeddings import process_image, preprocess, get_embedding, input_image_embedding


@pytest.fixture
def sample_image():
    """
    erstellt ein einfaches 224x224 RGB-testbild
    """
    img = Image.new('RGB', (224, 224), color='blue')
    return img


def test_process_image(sample_image):
    """
    testet, ob das bild korrekt verarbeitet wird
    """
    # wir benutzen bytesio, um das bild sample_image als JPEG zu speichern, ohne eine datei auf der festplatte zu erstellen - 
    # aus test gründen ;)
    img_bytes = BytesIO()
    sample_image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    processed_img = process_image(img_bytes)
    assert processed_img.mode == "RGB"
    assert isinstance(processed_img, Image.Image)


def test_preprocess(sample_image):
    """
    testet, ob die preprocessing-funktion ein tensor-format ausgibt
    """
    tensor = preprocess(sample_image)
    assert isinstance(tensor, torch.Tensor)
    # Batch-Dimension und 3 farbkanäle des tensors
    assert tensor.shape == (1, 3, 224, 224)


def test_get_embedding(sample_image):
    """
    testet die embedding-funktion mit ResNet18
    """
    device = torch.device("cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    
    embedding = get_embedding(sample_image, model, device)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == 512


def test_input_image_embedding(sample_image, monkeypatch):
    """
    testet die funktion input_image_embedding mit einem RGB-Bild
    """
    # sogenanntes "Monkeypatching", um sicherzustellen, dass das Bild nicht erneut geöffnet wird als debugging maßnahme
    monkeypatch.setattr("embeddings.process_image", lambda x: x)
    embedding = input_image_embedding(sample_image)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == 512 



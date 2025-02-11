import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from embeddings import process_image, preprocess, get_embedding, input_image_embedding


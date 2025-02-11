import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

def process_image(image):
    """
    Processes an image by resizing and converting it to RGB.
    """
    image = Image.open(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return image

def preprocess(image):
    """
    Preprocess the given image to the format required by the model.
    """
    
    # Define the image transformations (resize, center crop, and normalization)
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
        
    preprocessed_img = transform(image)
        
    return preprocessed_img.unsqueeze(0)

    
def get_embedding(image, model, device):
    """
    Extract the embedding of an image tensor using the pre-trained resnet18 model and return a numpy array.
    """
    img_tensor = preprocess(image).to(device)

    with torch.no_grad():
        feature_vector = model(img_tensor)
        return feature_vector.cpu().numpy().flatten().astype(np.float32)

    
def input_image_embedding(image):

    image = process_image(image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval().to(device)
    
    return get_embedding(image, model, device)
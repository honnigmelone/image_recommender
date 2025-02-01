from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def euclidean_distance(v1, v2):
    return distance.euclidean(v1, v2)

def manhatten_distance(v1, v2):
    return distance.cityblock(v1, v2)

def cos_similarity(v1, v2):
    return cosine_similarity([v1], [v2])[0][0]


def preprocess(img):
    """
    Preprocess the given image to the format required by the model.
    """
    
    try:
        # Define the image transformations (resize, center crop, and normalization)
        transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        

        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        preprocessed_img = transform(img)
        
        return preprocessed_img.unsqueeze(0)
    
    except Exception as e:
        print(f"Error while loading {img}")
        return None



def get_embedding(img, device):
    """
    Extract the embedding of an image tensor using the pre-trained resnet50 model and return a numpy array.
    """
    
    # Load a pre-trained ResNet model and replace the final layer
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()  # Replace the fully connected layer with an identity layer
    model.eval() # Set the model to evaluation mode

    with torch.no_grad():
        model.to(device)
        img = preprocess(img).to(device)
        return model(img).cpu().numpy().flatten()
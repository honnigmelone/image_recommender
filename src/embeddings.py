import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import pickle
from tqdm import tqdm

CHECKPOINT_INTERVAL = 10000

def load_data(path):
    """
    Load the data from the pickle file.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
    
def load_existing_embeddings(path):
    """
    Load the existing embeddings from the pickle file.
    """
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return []

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
        print(f"Error while loading {img}: {e}")
        return None
    
def get_embedding(img, model, device):
    """
    Extract the embedding of an image tensor using the pre-trained resnet50 model and return a numpy array.
    """
    
    with torch.no_grad():
        img_tensor = preprocess(img)
        if img_tensor is None:
            return None
        return model(img_tensor.to(device)).cpu().numpy().flatten()
    
def input_image_embedding(img_path):

    img = Image.open(img_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval().to(device)
    

    # Get the embedding
    embedding = get_embedding(img, model, device)
    return embedding
    
    
def generate_embeddings(imgdata_path, output_path, device):
    """Generate and save embeddings, loading from checkpoint if available."""
    
    # Load image metadata
    image_data = load_data(imgdata_path)

    # Load pre-trained model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval().to(device)

    # Load existing embeddings (checkpoint)
    embeddings = load_existing_embeddings(output_path)
    processed_ids = {entry["image_id"] for entry in embeddings}

    for i, data in enumerate(tqdm(image_data, desc="Generating Embeddings")):
        image_id, root, file = data["image_id"], data["root"], data["file"]

        # Skip already processed images
        if image_id in processed_ids:
            continue

        img_path = os.path.join(root, file)
        try:
            img = Image.open(img_path)
            embedding = get_embedding(img, model, device)
            if embedding is not None:
                embeddings.append({"image_id": image_id, "embedding": embedding})

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

        # Save checkpoint every 10,000 entries
        if len(embeddings) % CHECKPOINT_INTERVAL == 0:
            with open(output_path, "wb") as f:
                pickle.dump(embeddings, f)
            print(f"Checkpoint saved: {len(embeddings)} images processed.")

    # Final save
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Final embeddings saved to {output_path}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generate_embeddings("data/image_data.pkl", "data/embeddings.pkl", device)
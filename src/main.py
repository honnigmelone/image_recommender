import pickle
import os
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from PIL import Image
from config import IMAGE_DATA_OUTPUT_PATH, FEATURE_DATA_OUTPUT_PATH
from hashes import get_ahash, get_dhash
from histograms import get_histogram
from embeddings import get_embedding


CHECKPOINT = 2000


def main():

    with open(IMAGE_DATA_OUTPUT_PATH, "rb") as f:
        image_data = pickle.load(f)

    feature_data = []


    if os.path.exists(FEATURE_DATA_OUTPUT_PATH):
        with open(FEATURE_DATA_OUTPUT_PATH, "rb") as f:
            feature_data = pickle.load(f)

        processed_ids = {entry["image_id"] for entry in feature_data}
    
    else:
        processed_ids = set()

    # Define model for embeddings
    # Load pre-trained model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval().to(device)



    for images in tqdm(image_data, desc="Processing images"):
        image_id = images["image_id"]
        filepath = os.path.join(images["root"], images["file"])


        if image_id in processed_ids:
            continue
        
        try:

            with Image.open(filepath) as image:

                if image.mode != "RGB":
                    image = image.convert("RGB")


                embeddings = get_embedding(image, model, device)
                rgb_hists = get_histogram(image)
                ahashes = get_ahash(image)
                dhashes = get_dhash(image)


            feature_data.append({"image_id": image_id,"embeddings":embeddings, "rgb_hists": rgb_hists, "ahashes": ahashes, "dhashes": dhashes})

        except Exception as e:
            print(f"Error processing image {image_id}: {e}")



        if len(feature_data) % CHECKPOINT == 0:
            with open(FEATURE_DATA_OUTPUT_PATH, "wb") as f:
                pickle.dump(feature_data, f)

            print(f"Checkpoint reached. Saved {len(feature_data)} embeddings to pickle file.")


    with open(FEATURE_DATA_OUTPUT_PATH, "wb") as f:
        pickle.dump(feature_data, f)
        print(f"Saved remnaining {len(feature_data)} embeddings to pickle file.")

if __name__ == "__main__":
    main()
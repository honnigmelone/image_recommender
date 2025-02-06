import pickle
import os
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from PIL import Image
from config import IMAGE_DATA_OUTPUT_PATH, FEATURE_DATA_OUTPUT_PATH
from phashes import get_phash
from histograms import get_histogram
from embeddings import get_embedding


CHECKPOINT = 5000


def main():

    with open(IMAGE_DATA_OUTPUT_PATH, "rb") as f:
        image_data = pickle.load(f)

    feature_data = []


    if os.path.exists(FEATURE_DATA_OUTPUT_PATH):
        with open(os.path.join(FEATURE_DATA_OUTPUT_PATH), "rb") as f:
            processed_ids = {entry["image_id"] for entry in pickle.load(f)}

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
                hists = get_histogram(image)
                phashes = get_phash(image)


            feature_data.append({"image_id": image_id,"embeddings":embeddings, "colors": hists, "phashes": phashes})

        except Exception as e:
            print(f"Error processing image {image_id}: {e}")



        if len(feature_data) % CHECKPOINT == 0:
            with open(FEATURE_DATA_OUTPUT_PATH, "wb") as f:
                pickle.dump(feature_data, f)
            feature_data.clear()

            print(f"Checkpoint reached. Saved {len(feature_data)} embeddings to pickle file.")


    with open(FEATURE_DATA_OUTPUT_PATH, "wb") as f:
        pickle.dump(feature_data, f)
        print(f"Saved remnaining {len(feature_data)} embeddings to pickle file.")

if __name__ == "__main__":
    main()
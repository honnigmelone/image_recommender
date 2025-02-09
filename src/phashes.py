import cv2
import numpy as np
from PIL import Image
import pickle 
import os
from tqdm import tqdm
from config import IMAGE_DATA_OUTPUT_PATH, DATA_ROOT


def get_phash(image, hash_size=32):
    """
    Generates a perceptual hash (pHash) for an image.
    Returns a binary hash as a NumPy array.
    """
    image = np.array(image)
    # Resize and convert to grayscale
    image = cv2.resize(image, (hash_size, hash_size))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Compute mean pixel value
    mean = np.mean(gray)

    # Convert to binary hash
    phash = np.where(gray >= mean, 1, 0).flatten()

    return phash  # Returns the hash as a binary array

def input_image_phash(img_path):
    """
    Computes the pHash for an input image.
    """
    image = Image.open(img_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    return get_phash(image)

###########################################################################################

def generate_phases(image_data_path, output_path):
    """
    Generates the phase spectrum for each image in the database.
    """
    existing_phases = []
    with open(image_data_path, "rb") as f:
        data = pickle.load(f)
    if os.path.exists(output_path):

        with open(output_path, "rb") as f:
            existing_phases = pickle.load(f)
    
    processed_ids = {entry["image_id"] for entry in existing_phases}

    for image in tqdm(data):
        image_id = image["image_id"]
        filepath = os.path.join(image["root"], image["file"])

        if image_id in processed_ids:
            continue
        try:

            image = Image.open(filepath)
            if image.mode != "RGB":
                image = image.convert("RGB")

            phashes = get_phash(image)

            existing_phases.append({"image_id": image_id, "phashes": phashes})

        except Exception as e:
            print(f"Error processing image {filepath} : {e}")

        if len(existing_phases) % 1000 == 0:
            with open(output_path, "wb") as f:
                pickle.dump(existing_phases, f)

    with open(output_path, "wb") as f:
        pickle.dump(existing_phases, f)

if __name__ == "__main__":
    generate_phases(IMAGE_DATA_OUTPUT_PATH, os.path.join(DATA_ROOT, "phashes.pkl"))

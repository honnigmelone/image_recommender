import cv2
import numpy as np
import pickle
import os
from tqdm import tqdm
from PIL import Image
from config import IMAGE_DATA_OUTPUT_PATH, DATA_ROOT

# checkpoint size, in case it crashes :(
CHECKPOINT_INTERVAL = 1000


def get_histogram(image, bins=(64)):
    """
    Calculates and normalizes the color histogram of an image
    """
    image = np.array(image)
    
    red_hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
    green_hist = cv2.calcHist([image], [1], None, [bins], [0, 256])
    blue_hist = cv2.calcHist([image], [2], None, [bins], [0, 256])

    red_hist /= red_hist.sum()
    green_hist /= green_hist.sum()
    blue_hist /= blue_hist.sum()

    np_hist = np.concatenate((red_hist, green_hist, blue_hist), axis=0).reshape(-1)
    return np_hist


def input_image_histogram(img_path):
    """
    Computes the histogram for an input image.
    """
    image = Image.open(img_path)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return get_histogram(image)

##############################################################################################################

def load_data(path):
    """
    Loads the data from the pickle file.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
    
def load_existing_histograms(path):
    """
    Loads the existing histograms from the pickle file.
    """
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return []

def generate_histograms(imgdata_path, output_path):
    """
    Generates and saves color histograms, loading from the last checkpoint if there is one available.
    """
    image_data = load_data(imgdata_path)
    histograms = load_existing_histograms(output_path)
    processed_ids = {entry["image_id"] for entry in histograms}
    
    for i, data in enumerate(tqdm(image_data, desc="Generating Histograms")):
        image_id, root, file = data["image_id"], data["root"], data["file"]
        
        if image_id in processed_ids:
            continue
        
        img_path = os.path.join(root, file)
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        histogram = get_histogram(image)
        histograms.append({"image_id": image_id, "colors": histogram})
        
        if len(histograms) % CHECKPOINT_INTERVAL == 0:
            with open(output_path, "wb") as f:
                pickle.dump(histograms, f)
            print(f"Checkpoint saved: {len(histograms)} histograms processed.")
    
    with open(output_path, "wb") as f:
        pickle.dump(histograms, f)
    print(f"Final histograms saved to {output_path}")

if __name__ == "__main__":
    generate_histograms(IMAGE_DATA_OUTPUT_PATH, os.path.join(DATA_ROOT, "histograms.pkl"))
import cv2
import numpy as np
import pickle
import os
from tqdm import tqdm

# checkpoint size, in case it crashes :(
CHECKPOINT_INTERVAL = 10000

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

def load_image(image_path):
    """
    Loads an image from the given path and returns it as a RGB array.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.array(image_rgb)

def calculate_histogram(image, bins=(8, 8, 8)):
    """
    Calculates and normalizes the color histogram of an image by flattening it.
    """
    histogram = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(histogram, histogram)
    return histogram.flatten()

def input_image_histogram(img_path):
    """
    Computes the histogram for an input image.
    """
    image = load_image(img_path)
    if image is None:
        return None
    return calculate_histogram(image)

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
        image = load_image(img_path)
        if image is None:
            print(f"Error loading image: {img_path}")
            continue
        
        histogram = calculate_histogram(image)
        histograms.append({"image_id": image_id, "histogram": histogram})
        
        if len(histograms) % CHECKPOINT_INTERVAL == 0:
            with open(output_path, "wb") as f:
                pickle.dump(histograms, f)
            print(f"Checkpoint saved: {len(histograms)} histograms processed.")
    
    with open(output_path, "wb") as f:
        pickle.dump(histograms, f)
    print(f"Final histograms saved to {output_path}")

if __name__ == "__main__":
    generate_histograms("data/image_data.pkl", "data/histograms.pkl")
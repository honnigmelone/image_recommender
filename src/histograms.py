import cv2
import numpy as np
import pickle
import os
from tqdm import tqdm
from PIL import Image
from config import IMAGE_DATA_OUTPUT_PATH, DATA_ROOT

# checkpoint size, in case it crashes :(
CHECKPOINT_INTERVAL = 1000


def get_histogram(image, bins=[8, 8, 8]):
    """
    Calculates and normalizes the color histogram of an image
    """
    image = np.array(image)
    histogram = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()

    return histogram

def get_hsv_histogram(image, bins=[8, 8, 8]):
    """
    Calculates and normalizes the HSV color histogram of an image.
    """
    image = np.array(image)  # Convert image to NumPy array
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # Convert to HSV color space

    # Compute the 3D histogram (H, S, V channels)
    histogram = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

    # Normalize the histogram so that the values sum to 1
    histogram = cv2.normalize(histogram, histogram).flatten()

    return histogram

def input_image_hsv_histogram(image):
    """
    Computes the HSV histogram for an input image.
    """
    image = Image.open(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
       
    cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    
    return get_hsv_histogram(image)



def input_image_histogram(image):
    """
    Computes the histogram for an input image.
    """
    image = Image.open(image)

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
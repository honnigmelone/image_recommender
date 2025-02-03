import pickle
import numpy as np 
import os
from tqdm import tqdm
from PIL import Image
from config import PROJECT_ROOT



# Define the function to create a sprite image from a batch of images.
def create_sprite(data):
    """
    Tile images into a single sprite image.
    Handles padding automatically if the number of images is not a perfect square.
    
    Args:
        data (numpy array): Array of shape (num_images, height, width, channels)
    
    Returns:
        sprite (numpy array): Large sprite image containing all input images.
    """
    
    # If images are grayscale, convert to RGB.
    if len(data.shape) == 3:  # (num_images, height, width) -> Grayscale
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))  # Now shape is (num_images, height, width, 3)

    num_images = data.shape[0]
    n = int(np.ceil(np.sqrt(num_images)))  # grid size (n x n)

    # Pad images if necessary.
    padding = ((0, n ** 2 - num_images), (0, 0), (0, 0), (0, 0))
    data = np.pad(data, padding, mode='constant', constant_values=0)

    # Reshape and tile images into a grid.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3, 4))
    data = data.reshape((n * data.shape[1], n * data.shape[3], 3))
    return data

def create_metadata_file(image_data, metadata_path="logs/metadata.tsv"):
    """
    Create a metadata file containing image paths.
    
    Args:
        image_paths (list): List of image paths
        metadata_path (str): Path to save metadata file
    """
    with open(metadata_path, 'w') as f:

        for entry in image_data:
            f.write(f"{entry['image_id']}\n")


def load_image_data(image_data_path):
    with open(image_data_path, 'rb') as f:
        image_data = pickle.load(f)
    return image_data


def process_images(image_paths, image_size=(32, 32)):
    """
    Load and preprocess images from given paths.
    Args:
        image_paths (list): List of image file paths.
        image_size (tuple): Desired size of the images (width, height).
    Returns:
        np.array: Array of preprocessed image data.
    """
    image_arrays = []
    for path in tqdm(image_paths):
        try:
            img = Image.open(path).resize(image_size)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_array = np.array(img)
            image_arrays.append(img_array)
        except Exception as e:
            print(f"Skipping {path}: {e}")
    return np.stack(image_arrays)



if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    image_data_path = os.path.join(PROJECT_ROOT, r"data\image_data.pkl")

    image_data = load_image_data(image_data_path)

    # Create metadata file
    create_metadata_file(image_data, metadata_path=os.path.join(PROJECT_ROOT, r"logs/metadata.tsv"))

    # Map image IDs to paths
    image_id_to_path = {entry["image_id"]: os.path.join(entry["root"], entry["file"]) for entry in image_data}
    image_paths = [image_id_to_path[entry["image_id"]] for entry in image_data]


    image_data_array = process_images(image_paths)

    # Create sprite image
    sprite_image = create_sprite(image_data_array)
    # Convert NumPy array to PIL Image and save
    sprite_pil = Image.fromarray(sprite_image.astype(np.uint8))
    sprite_pil.save(os.path.join(PROJECT_ROOT, r"logs/sprite.png"))

    print("Sprite image saved as logs/sprite.png")
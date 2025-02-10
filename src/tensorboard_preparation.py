import pickle
import numpy as np 
import os
from tqdm import tqdm
from PIL import Image
from config import IMAGE_DATA_OUTPUT_PATH, FEATURE_DATA_OUTPUT_PATH, LOGS_ROOT
from tensorboard.plugins import projector
import tensorflow as tf



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

def create_metadata_file(image_data, metadata_path):
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


def process_images(image_paths, image_size=(12, 12)):
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

def prepare_tensorboard_data(mode):

    # Remove old checkpoint files so we always create fresh ones
    for filename in os.listdir(LOGS_ROOT):
        if filename.startswith("embedding.ckpt") or filename == "checkpoint":
            file_path = os.path.join(LOGS_ROOT, filename)
            os.remove(file_path)

    # Load the embeddings data
    with open (FEATURE_DATA_OUTPUT_PATH, 'rb') as f:
        embeddings_data = pickle.load(f)
    embeddings_array = np.array([entry[mode] for entry in embeddings_data])

    # Create a variable to hold the embeddings
    embedding_var = tf.Variable(embeddings_array, name='embedding')

    # Create and save a checkpoint for the embedding
    checkpoint = tf.train.Checkpoint(embedding=embedding_var)
    checkpoint.save(os.path.join(LOGS_ROOT, 'embedding.ckpt'))

    # Set up projector config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'

    # Add path to the sprite image
    embedding.sprite.image_path = 'sprite.png'
    embedding.sprite.single_image_dim.extend([12, 12])
    projector.visualize_embeddings(LOGS_ROOT, config)
    print(f'Projector config saved at: {LOGS_ROOT}')

def main(mode):
    
    os.makedirs("logs", exist_ok=True)


    image_data = load_image_data(IMAGE_DATA_OUTPUT_PATH)

    # Create metadata file
    metadata_path=os.path.join(LOGS_ROOT, "metadata.tsv")
    if not os.path.exists(metadata_path):
        create_metadata_file(image_data, metadata_path)

    # Map image IDs to paths
    image_id_to_path = {entry["image_id"]: os.path.join(entry["root"], entry["file"]) for entry in image_data}
    image_paths = [image_id_to_path[entry["image_id"]] for entry in image_data]


    # Create sprite image
    sprite_image_path = os.path.join(LOGS_ROOT, "sprite.png")

    if not os.path.exists(sprite_image_path):
        image_data_array = process_images(image_paths)
        sprite_image = create_sprite(image_data_array)
        # Convert NumPy array to PIL Image and save
        sprite_pil = Image.fromarray(sprite_image.astype(np.uint8))
        sprite_pil.save(os.path.join(LOGS_ROOT, "sprite.png"))

        print("Sprite image saved as logs/sprite.png")
    prepare_tensorboard_data(mode)

if __name__ == "__main__":
    #main("embeddings")
    main("rgb_hists")
    
    # Run the tensorboard command in the terminal
    # tensorboard --logdir logs/

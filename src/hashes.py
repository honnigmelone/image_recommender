import numpy as np
from PIL import Image
import imagehash

def process_image(image):
    """
    Processes an image by resizing and converting it to RGB.
    """

    image = Image.open(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return image


def get_ahash(image, hash_size=32):
    """
    Generates a perceptual hash (pHash) for an image.
    Returns a binary hash as a NumPy array.
    """
    # Open and preprocess image
    image = image.convert("L")
    
    # Compute perceptual hash
    phash = imagehash.average_hash(image, hash_size=hash_size)

    # convert to 0 and 1
    hash_array = phash.hash.astype(np.uint8).flatten()
    
    return hash_array

def get_dhash(image, hash_size=32):
    """
    Generates a perceptual hash (pHash) for an image.
    Returns a binary hash as a NumPy array.
    """
    # Open and preprocess image
    image = image.convert("L")
    
    # Compute perceptual hash
    dhash = imagehash.dhash(image, hash_size=hash_size)

    # convert to 0 and 1
    hash_array = dhash.hash.astype(np.uint8).flatten()
    
    return hash_array


def input_image_ahash(image):
    """
    Computes the pHash for an input image.
    """
    image = process_image(image)

    return get_ahash(image)

def input_image_dhash(image):
    """
    Computes the dHash for an input image.
    """
    image = process_image(image)

    return get_dhash(image)

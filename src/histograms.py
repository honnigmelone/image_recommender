import cv2
import numpy as np
from PIL import Image

def process_image(image):
    """
    Processes an image by resizing and converting it to RGB.
    """
    image = Image.open(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return image


def get_histogram(image, bins=[8, 8, 8]):
    """
    Calculates and normalizes the color histogram of an image
    """
    image = np.array(image)
    histogram = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    histogram = cv2.normalize(histogram, histogram, norm_type=cv2.NORM_L1).flatten()

    return histogram.astype(np.float32)


def input_image_histogram(image):
    """
    Computes the histogram for an input image.
    """
    image = process_image(image)

    return get_histogram(image)

from skimage.feature import hog
import numpy as np
import cv2


def get_hog_features(image):
    """
    Extracts HOG features from an image.
    """
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    return features  # 1D feature vector
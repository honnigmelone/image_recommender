import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
from config import FEATURE_DATA_OUTPUT_PATH
from sklearn.metrics.pairwise import cosine_similarity
from database import select_image_from_database
from histograms import input_image_histogram
from hashes import input_image_ahash, input_image_dhash
from embeddings import input_image_embedding

FEATURE_DATA_CACHE = None  


def plot_similar_images(input_images, top_similarities, target_size=(512,512)):

    # Combine images into a single image for visualization
    combined_width = target_size[0] * (len(input_images))
    combined_image = Image.new("RGB", (combined_width, target_size[1]))

    for i, image in enumerate(input_images):
        img = Image.open(image).resize(target_size).convert("RGB")
        combined_image.paste(img, (i * target_size[0], 0))

    # Plot the combined image and the top similar images
    fig, axs = plt.subplots(1, len(top_similarities)+2, figsize=(15,5), gridspec_kw={'width_ratios': [1.5, 0.01] + [1] * len(top_similarities)})
    axs[0].imshow(combined_image)
    axs[0].axis("off")
    axs[0].set_title("Input Image")

    axs[1].axis("off")
    axs[1].add_patch(patches.Rectangle((0,0), 1, 1, color = "black"))
    
    for i, (_, sim_score, filepath) in enumerate(top_similarities):
        img = Image.open(filepath).resize(target_size)
        axs[i+2].imshow(img)
        axs[i+2].axis("off")
        axs[i+2].set_title(f"Sim_score: {sim_score:.2f}")

    plt.tight_layout()
    plt.show()

def load_feature_data():
    """Loads feature data globally so it stays in memory across function calls."""
    global FEATURE_DATA_CACHE  # Access the global variable
    
    # Use the cached data if it exists
    if FEATURE_DATA_CACHE is not None:
        return FEATURE_DATA_CACHE

    with open(FEATURE_DATA_OUTPUT_PATH, "rb") as f:
        FEATURE_DATA_CACHE = pickle.load(f)  # Store in memory
    
    return FEATURE_DATA_CACHE


def calculate_similarities(input_images, cursor, mode, metric, top_k=5, verbose=False):
    global FEATURE_DATA_CACHE  # Access the global variable
    mode = mode.lower()
    start_time = time.time()

    # Load feature data with caching
    load_time_start = time.time()
    

    data = load_feature_data()
    
    load_time_end = time.time()
    
    if verbose:
        print(f"Time to load feature data: {load_time_end - load_time_start:.4f} second")

    feature_extraction_start = time.time()

    # Extract features based on the mode
    if mode == "rgb":
        input_transformed = np.mean([input_image_histogram(image) for image in input_images], axis=0)
        X = np.array([entry["rgb_hists"] for entry in data])
    elif mode == "embeddings":
        input_transformed = np.mean([input_image_embedding(image) for image in input_images], axis=0)
        X = np.array([entry["embeddings"] for entry in data])
    elif mode == "ahashes":
        input_transformed = np.mean([input_image_ahash(image) for image in input_images], axis=0)
        X = np.array([entry["ahashes"] for entry in data])
    elif mode == "dhashes":
        input_transformed = np.mean([input_image_dhash(image) for image in input_images], axis=0)
        X = np.array([entry["dhashes"] for entry in data])
    else:
        raise ValueError("Invalid mode. Choose from 'rgb', 'embeddings', 'ahashes', 'dhashes'.")

    feature_extraction_end = time.time()

    if verbose:
        print(f"Time to extract features: {feature_extraction_end - feature_extraction_start:.4f} seconds")

    similarity_computation_start = time.time()

    # Compute similarity or distance
    if metric == "cosine":
        input_vec = input_transformed.reshape(1, -1)
        sims = cosine_similarity(input_vec, X).flatten()
        top_k_indices = np.argpartition(-sims, top_k)[:top_k]
        top_k_indices = top_k_indices[np.argsort(-sims[top_k_indices])]

    elif metric == "euclidean":
        sims = np.linalg.norm(X - input_transformed, axis=1)
        top_k_indices = np.argpartition(sims, top_k)[:top_k]
        top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])]

    elif metric == "manhattan":
        sims = np.sum(np.abs(X - input_transformed), axis=1)
        top_k_indices = np.argpartition(sims, top_k)[:top_k]
        top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])]

    elif metric == "hamming":
        sims = np.sum(input_transformed != X, axis=1)
        top_k_indices = np.argpartition(sims, top_k)[:top_k]
        top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])]

    else:
        raise ValueError("Invalid metric! Use 'cosine', 'euclidean', or 'manhattan'.")

    similarity_computation_end = time.time()

    if verbose:
        print(f"Time to compute similarities: {similarity_computation_end - similarity_computation_start:.4f} seconds")

    retrieval_start = time.time()

    # Retrieve top-k similar images
    top_similarities = []
    for i in top_k_indices:
        image_id = data[i]["image_id"]
        sim_value = sims[i]
        file_path = select_image_from_database(image_id, cursor)
        if file_path:
            top_similarities.append((image_id, sim_value, file_path))

    retrieval_end = time.time()
    if verbose:
        print(f"Time to retrieve top-K images: {retrieval_end - retrieval_start:.4f} seconds")

    total_time = time.time() - start_time
    if verbose:
        print(f"Total execution time: {total_time:.4f} seconds")


    # Plot the similar images
    plot_similar_images(input_images, top_similarities)
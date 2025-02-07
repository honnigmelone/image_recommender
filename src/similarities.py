import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches 
from config import FEATURE_DATA_OUTPUT_PATH
from sklearn.metrics.pairwise import cosine_similarity
from database import select_image_from_database
from embeddings import input_image_embedding
from histograms import input_image_histogram
from phashes import input_image_phash

CACHED_DATA = None


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

def calculate_similarities(input_images, mode, cursor, top_k=5, metric="cosine"):

    mode = mode.lower()

    with open(FEATURE_DATA_OUTPUT_PATH, "rb") as f:
        data = pickle.load(f)

    if mode == "color":
        input_transformed = np.mean([input_image_histogram(image) for image in input_images], axis=0)
        X = np.array([entry["colors"] for entry in data])
    elif mode == "embeddings":
        input_transformed = np.mean([input_image_embedding(image) for image in input_images], axis=0)
        X = np.array([entry["embeddings"] for entry in data])
    elif mode == "phashes":
        input_transformed = np.mean([input_image_phash(image) for image in input_images], axis=0)
        X = np.array([entry["phashes"] for entry in data])

    else:
        raise ValueError("Invalid input :O  You must choose between color, embeddings and phashes.")

    
    # Input shape (D,)
    # Compute similarity or distance in a vectorized manner.
    if metric == "cosine":
        input_vec = input_transformed.reshape(1, -1)  # shape (1, D)
        sims = cosine_similarity(input_vec, X).flatten()  # shape (N,)
        top_k_indices = np.argpartition(-sims, top_k)[:top_k]  # Select top-K largest (negated for descending order)
        top_k_indices = top_k_indices[np.argsort(-sims[top_k_indices])]  # Sort descending

    # Vectorized distance with numpy
    elif metric == "euclidean":
        sims = np.linalg.norm(X - input_transformed, axis=1)
        top_k_indices = np.argpartition(sims, top_k)[:top_k]  # Select top-K smallest
        top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])]  # Sort ascending

    elif metric == "manhattan":
        sims = np.sum(np.abs(X - input_transformed), axis=1)
        top_k_indices = np.argpartition(sims, top_k)[:top_k]
        top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])]

    elif metric == "hamming":
        # Vectorized Hamming distance
        sims = np.sum(input_transformed != X, axis=1)  # Lower distance = more similar
        top_k_indices = np.argpartition(sims, top_k)[:top_k]
        top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])]

    else:
        raise ValueError("Invalid metric! Use 'cosine', 'euclidean', or 'manhattan'.")

    # Retrieve top-k similar images
    top_similarities = []
    for i in top_k_indices:
        image_id = data[i]["image_id"]
        sim_value = sims[i]  # Similarity or distance value
        file_path = select_image_from_database(image_id, cursor)

        if file_path:
            top_similarities.append((image_id, sim_value, file_path))

    plot_similar_images(input_images, top_similarities)

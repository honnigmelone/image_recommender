import pickle
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches 
from config import PROJECT_ROOT
from sklearn.metrics.pairwise import cosine_similarity
from database import select_image_from_database
from embeddings import input_image_embedding

def plot_similar_images(input_image, top_similarities, target_size=(512,512)):
    
    img = Image.open(input_image).resize(target_size)

    fig, axs = plt.subplots(1, len(top_similarities)+2, figsize=(15,5), gridspec_kw={'width_ratios': [1.5, 0.01] + [1] * len(top_similarities)})
    axs[0].imshow(img)
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

def calculate_similarities(input_image, mode, top_k=5, metric="cosine"):

    mode = mode.lower()

    if mode == "color":
        pkl_path = os.path.join(PROJECT_ROOT, r"data/color.pkl")
        #input_transformed = get_hist(input_image)
    elif mode == "embeddings":
        pkl_path = os.path.join(PROJECT_ROOT, r"data/embeddings.pkl")
        input_transformed = input_image_embedding(input_image)
    elif mode == "phases":
        pkl_path = os.path.join(PROJECT_ROOT, r"data/phases.pkl")
        #input_transformed = get_phases(input_image)
    else:
        raise ValueError("Invalid input :O  You must choose between color, embeddings and phases.")
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    
    X = np.array([next(value for key, value in entry.items() if key != "image_id") for entry in data])
    # input shape (D,)
    # Compute similarity or distance in a vectorized manner.
    if metric == "cosine":
        input_vec = input_transformed.reshape(1, -1)  # shape (1, D)
        sims = cosine_similarity(input_vec, X).flatten()  # shape (N,)
        sorted_sims = np.argsort(sims)[::-1]  # Higher = more similar.
    # Vectorized distance with numpy
    elif metric == "euclidean":
        sims = np.linalg.norm(X - input_transformed, axis=1)
        sorted_sims = np.argsort(sims)  # Lower distance = more similar.
    elif metric == "manhattan":
        sims = np.sum(np.abs(X - input_transformed), axis=1)
        sorted_sims = np.argsort(sims)  # Lower distance = more similar.

    else:
        raise ValueError("Invalid metric! Use 'cosine', 'euclidean', or 'manhattan'.")

    # Retrieve top-k similar images
    top_similarities = []
    for i in sorted_sims[:top_k]:
        image_id = data[i]["image_id"]
        sim_value = sims[i]  # Similarity or distance value
        file_path = select_image_from_database(image_id)

        if file_path:
            top_similarities.append((image_id, sim_value, file_path))

    plot_similar_images(input_image, top_similarities)
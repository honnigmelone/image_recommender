# Image Recommender System

[![GitHub Repo Size](https://img.shields.io/github/repo-size/honnigmelone/image_recommender.svg)](https://github.com/honnigmelone/image_recommender)  [![GitHub License](https://img.shields.io/github/license/honnigmelone/image_recommender.svg)](LICENSE)  [![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/honnigmelone/image_recommender/main.yml)](https://github.com/honnigmelone/image_recommender/actions)  [![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

Welcome to the **Image Recommender System**, a robust platform for generating image recommendations based on visual similarity. In short, you can:

- Generate image metadata and load it into a SQL database.
- Extract image features such as neural network embeddings, RGB color histograms, and hashes (ahashes and dhashes).
- Input an image and retrieve the top-k closest images from your database using similarity metrics like cosine similarity, Euclidean distance, and Hamming distance.
- Optionally, reduce dimensions and visualize the image distribution using TensorBoard or custom plots.

Example similarity search outputs:

**Resnet18 embeddings:**
![image](https://github.com/user-attachments/assets/031fc9d5-dbed-4042-b671-c129f37fd10c)
**Rgb color histograms:**
![image](https://github.com/user-attachments/assets/c42fe2e6-dd17-4f12-926f-851351124716)
**Average hashes:**
![image](https://github.com/user-attachments/assets/13dcc329-f85e-434f-b19f-260571dd0f4e)



---

## Table of Contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Visualisation](#visualisation)
- [Need Help?](#need-help)
- [License](#license)
- [Authors](#authors)

---

## How It Works

The Image Recommender System consists of a modular pipeline with the following key components:

1. **Metadata Generation & Database Storage**  
   - **Image Metadata Extraction:** The system scans your dataset to extract metadata (e.g., image IDs, file paths) and prepares it for database insertion.
   - **SQL Database Integration:** The extracted metadata is loaded into a SQL database for efficient querying and management.

2. **Feature Extraction**  
   - **Neural Network Embeddings:**  
     Uses pre-trained models (e.g., ResNet18) to extract high-dimensional feature vectors that capture semantic content.
   - **RGB Color Histograms:**  
     Computes histograms to represent the dominant color distributions in images.
   - **Hash-Based Features:**  
     Generates hashes (ahashes and dhashes) that capture structural and textural information.

3. **Similarity Computation & Retrieval**  
   - **Distance Metrics:**  
     The system supports various metrics, including cosine similarity, Euclidean distance, Hamming distance, and Manhattan distance.
   - **Top-K Retrieval:**  
     Based on the selected similarity metric and mode, the system retrieves the top-k most similar images from the database.

4. **Visualization & Exploration**  
   - **Jupyter Notebook Interface:**  
     An interactive notebook (`show_similarites.ipynb`) allows you to input images, run similarity queries, and display the results.
   - **TensorBoard & Dimensionality Reduction:**  
     You can also reduce the feature dimensions (e.g., using UMAP) and visualize the embeddings with TensorBoard for an intuitive exploration of the dataset.

For more details, please refer to the [Documentation](Big_Data_Image_Recommender_Doku.pdf).

---

## Installation

To get started, follow these steps:

1. **Clone the repository:**
```
git clone https://github.com/honnigmelone/image_recommender.git
```

2. **Navigate to the project directory**
```
cd image_recommender/
```

3. **Install the dependencies**
```
pip install -r requirements.txt
```

---

## How to use

After setting up the repository, the pipeline runs in several stages:

1. **Configure Your Dataset**  
- Open the configuration file (`config.py`) and update the path to **YOUR** image dataset.  
- Other paths typically require no changes.

![image](https://github.com/user-attachments/assets/7c78e34f-bfdd-4652-a911-d38aa2778d22)

2. **Run the Generator and create the database, Run follwoing commands from directory root**

```
python src/generator.py
```

3. **Execute the main loop to extract feature data**
Execute the main pipeline to extract image features. (This step may take a while, depending on your dataset size.)

```
python src/main.py
```

4. **View Similarity Results**  
Open and run the Jupyter Notebook `show_similarites.ipynb`:
- Add the paths of your input images to the provided list.
- The notebook calls the `calculate_similarites()` function from `similarites.py`, which uses the following parameters:
  - **input_images:** A list of image paths.
  - **cursor:** A database cursor for efficient data retrieval.
  - **mode:** Similarity mode (choose from: embeddings, rgb, ahashes, dhashes).
  - **metric:** Similarity metric (options include cosine, euclidean, hamming, manhattan).
  - **top_k:** Number of similar images to retrieve (default is 5).
  - **verbose:** Set to `True` to display execution times (default is `False`).


---


## **Visualisation**

You can visualize the images either in a Tensorboard or with Dimensionreduction

### **Tensorboard**

1. **Prepare Visualization Data**  
Generate the metadata file, sprite image, and checkpoint data. You can specify the mode in the main function like "embeddings" or "rgb_hists":

```
python src/tensorboard_preparation.py
```

2. **Launch TensorBoard**  
Run the following command to start TensorBoard on your localhost:

```
tensorboard --logdir logs/
```

3. **Example visualisation using color similarities with UMAP(UMAP only takes first 5000 entries)**
![image](https://github.com/user-attachments/assets/d192ebb7-b4ff-4cb1-aa4f-f883d3d78a8f)


### **UMAP**

1. **Just open the jupyter Notebook ```umap_analysis.ipynb``` and play with the functions :)** 
You can specify the mode, and top_k which represents the cluster values
Please be aware that umap needs a lot of memory to run on large datasets
You can either reduce dimensions and cluster afterwards or the other way around depending on your use case!

## Need Help?

If you have any questions, encounter issues, or need assistance, please open an issue in the GitHub repository. We’re here to help and value your feedback!

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Authors

- [honnigmelone](https://github.com/honnigmelone)
- [heyitsalina](https://github.com/heyitsalina)
- [sekkurocode](https://github.com/sekkurocode)

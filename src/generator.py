import os
import pickle
from tqdm import tqdm
from config import PATH_TO_DIR, OUTPUT_PATH

#Generator that yields a tuple of the root directory, filename, size
def generator_to_pickle(path_to_dir):
    for root, _, files in os.walk(path_to_dir):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                size = os.path.getsize(os.path.join(root, file))
                yield root, file, size


def save_data_to_pickle(gen, output_path):
    image_id = 0
    image_metadata_list = []
    for root, file, size in tqdm(gen, desc="Processing"):
        image_metadata = {"image_id":image_id, "root": root, "file": file, "size": size}
        image_metadata_list.append(image_metadata)
        image_id += 1

    with open(output_path, "wb") as f:
        pickle.dump(image_metadata_list, f)


gen = generator_to_pickle(PATH_TO_DIR)
save_data_to_pickle(gen, OUTPUT_PATH)
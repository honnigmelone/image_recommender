import os
import pickle
import tensorflow as tf
import numpy as np
from config import PROJECT_ROOT, LOGS_ROOT
from tensorboard.plugins import projector

embeddings_path = os.path.join(PROJECT_ROOT, r"data\embeddings.pkl")

with open (embeddings_path, 'rb') as f:
    embeddings_data = pickle.load(f)
embeddings_array = np.array([entry["embedding"] for entry in embeddings_data])

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
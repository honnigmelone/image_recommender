from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity

def euclidean_distance(v1, v2):
    return distance.euclidean(v1, v2)

def manhatten_distance(v1, v2):
    return distance.cityblock(v1, v2)

def cos_similarity(v1, v2):
    return cosine_similarity([v1], [v2])[0][0]

